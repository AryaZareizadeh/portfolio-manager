"""Validated UTC spot candles; public downloads never require credentials.

CSV timestamps identify candle *open* times. A candle is available for features
only after its entire interval has elapsed. Missing candles remain missing.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
import hashlib
import io
import json
import os
from pathlib import Path
import re
import ssl
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import parse_qs, urlencode, urlparse
import zipfile

import numpy as np
import pandas as pd

from .timeframes import INTERVALS, TIMEFRAMES, interval_delta


DAY = pd.Timedelta(days=1)
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
ARCHIVE_BASE = "https://data.binance.vision/data/spot"
KUCOIN_CANDLES = "https://api.kucoin.com/api/v1/market/candles"
WINDOWS_DEVICES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                   *(f"LPT{i}" for i in range(1, 10))}
NETWORK_WORKERS = 4
_HTTP_SLOTS = threading.BoundedSemaphore(NETWORK_WORKERS)


def _progress(callback, coin, phase, completed=None, total=None, *, source=None, detail=None):
    """Emit primitive work counters on the coordinating thread, outside fetch catches."""
    if callback is not None:
        event = {"coin": coin, "phase": phase, "completed": completed, "total": total}
        if source is not None:
            event["source"] = source
        if detail is not None:
            event["detail"] = detail
        callback(event)


def _coins(coins: list[str]) -> list[str]:
    normalized = [str(coin).strip().upper() for coin in coins]
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("Select at least one unique coin")
    for coin in normalized:
        if not re.fullmatch(r"[A-Z0-9]{1,20}", coin) or coin == "USDT" or coin in WINDOWS_DEVICES:
            raise ValueError(f"Invalid coin {coin!r}; use base symbols such as BTC, excluding USDT")
    return normalized


def _utc(value=None) -> pd.Timestamp:
    timestamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Invalid timestamp")
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _timestamps(values: pd.Series) -> pd.Series:
    """Interpret each epoch independently: archives changed ms to us in 2025."""
    text = values.astype(str).str.strip()
    numeric_mask = text.str.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    if numeric_mask.any():
        numeric = pd.to_numeric(text[numeric_mask], errors="raise")
        magnitude = numeric.abs()
        for unit, mask in [("ns", magnitude >= 1e17),
                           ("us", (magnitude >= 1e14) & (magnitude < 1e17)),
                           ("ms", (magnitude >= 1e11) & (magnitude < 1e14)),
                           ("s", magnitude < 1e11)]:
            selected = numeric[mask]
            result.loc[selected.index] = pd.to_datetime(selected, unit=unit, utc=True, errors="coerce")
    if (~numeric_mask).any():
        result.loc[~numeric_mask] = pd.to_datetime(text[~numeric_mask], utc=True, format="mixed", errors="coerce")
    if result.isna().any():
        raise ValueError("Data contains invalid timestamps")
    return result


def _missing_dates(frame: pd.DataFrame, start=None, end=None) -> list[str]:
    if frame.empty and (start is None or end is None):
        return []
    left = frame.timestamp.min() if start is None else start
    right = frame.timestamp.max() + DAY if end is None else end
    expected = pd.date_range(left, right, freq="D", inclusive="left")
    missing = expected.difference(pd.DatetimeIndex(frame.timestamp))
    return missing.strftime("%Y-%m-%d").tolist()


def _interval_quality(frame: pd.DataFrame, interval: str, start=None, end=None) -> dict:
    """Count missing slots without materializing years of absent intraday rows."""
    delta = interval_delta(interval)
    if frame.empty and (start is None or end is None):
        return {"interval": interval, "rows": 0, "missing_count": 0,
                "missing_dates": [], "missing_dates_truncated": False}
    left = _utc(start) if start is not None else frame.timestamp.min()
    right = _utc(end) if end is not None else frame.timestamp.max() + delta
    actual = pd.DatetimeIndex(frame.loc[(frame.timestamp >= left) & (frame.timestamp < right), "timestamp"])
    actual = actual.sort_values().unique().as_unit("ns")
    expected_count = max(0, int((right - left) // delta))
    missing_count = max(0, expected_count - len(actual))
    edges = np.concatenate(([left.value - delta.value], actual.asi8, [right.value]))
    gap_indices = np.flatnonzero(np.diff(edges) > delta.value)
    missing_sample = []
    for index in gap_indices:
        count = min(100 - len(missing_sample), int((edges[index + 1] - edges[index]) // delta.value) - 1)
        if count:
            stamps = pd.date_range(pd.Timestamp(edges[index] + delta.value, tz="UTC"), periods=count, freq=delta)
            missing_sample.extend(stamp.isoformat() for stamp in stamps)
        if len(missing_sample) == 100:
            break
    return {"interval": interval, "rows": len(frame), "missing_count": missing_count,
            "missing_dates": missing_sample, "missing_dates_truncated": missing_count > len(missing_sample)}


def _normalize(frame: pd.DataFrame, coin: str, as_of: pd.Timestamp, interval: str = "1d") -> pd.DataFrame:
    delta = interval_delta(interval)
    frame = frame.rename(columns=lambda name: str(name).strip().lower())
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{coin}: missing CSV columns {sorted(missing)}")
    frame = frame[COLUMNS].copy()
    frame["timestamp"] = _timestamps(frame.timestamp)
    if (frame.timestamp != frame.timestamp.dt.floor(delta)).any():
        raise ValueError(f"{coin}: {interval} candle open timestamps must align to the UTC interval grid")
    if frame.timestamp.duplicated().any():
        raise ValueError(f"{coin}: duplicate candle dates")
    for column in COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if not np.isfinite(frame[COLUMNS[1:]].to_numpy()).all():
        raise ValueError(f"{coin}: OHLCV values must be finite")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame.volume < 0).any():
        raise ValueError(f"{coin}: prices must be positive and volume nonnegative")
    if ((frame.high < frame[["open", "close", "low"]].max(axis=1)) |
            (frame.low > frame[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError(f"{coin}: inconsistent OHLC values")
    complete = frame.timestamp + delta <= as_of
    incomplete = int((~complete).sum())
    frame = frame.loc[complete].sort_values("timestamp").reset_index(drop=True)
    quality = _interval_quality(frame, interval)
    # Preserve the complete daily date list for the original daily-only API.
    if interval == "1d":
        quality.update(missing_dates=_missing_dates(frame), missing_dates_truncated=False)
    quality["incomplete_dropped"] = incomplete
    frame.attrs["quality"] = quality
    return frame


def load_data(data_dir: Path | str, coins: list[str], as_of=None) -> dict[str, pd.DataFrame]:
    """Load validated local history without filling gaps or incomplete days.

    Additional-source CSVs use the same schema and USDT quote. Describe their
    source in COIN.metadata.json; absent provenance is recorded as user supplied.
    """
    directory = Path(data_dir)
    cutoff = _utc(as_of)
    frames = {}
    for coin in _coins(coins):
        path = directory / f"{coin}.csv"
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        frame = _normalize(pd.read_csv(io.BytesIO(payload)), coin, cutoff)
        if frame.empty:
            raise ValueError(f"{coin}: no completed daily candles available")
        sidecar = directory / f"{coin}.metadata.json"
        metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {
            "source": "local_csv_user_supplied", "quote_currency": "USDT"}
        if metadata.get("data_sha256") and metadata["data_sha256"] != digest:
            raise ValueError(f"{coin}: CSV hash differs from its provenance metadata")
        if "coin" in metadata and str(metadata["coin"]).strip().upper() != coin:
            raise ValueError(f"{coin}: provenance identifies a different coin")
        for quote_field in ("quote_currency", "quote"):
            if quote_field in metadata and str(metadata[quote_field]).strip().upper() != "USDT":
                raise ValueError(f"{coin}: source quote currency must be USDT")
        if "interval" in metadata and str(metadata["interval"]).strip() != "1d":
            raise ValueError(f"{coin}: source interval must be 1d")
        if "symbol" in metadata:
            symbol = str(metadata["symbol"]).strip().upper()
            binance_source = str(metadata.get("source", "")).strip().lower() == "binance spot"
            simple_usdt_pair = re.fullmatch(r"[A-Z0-9]+USDT", symbol) is not None
            kucoin_source = str(metadata.get("source", "")).strip().casefold() == "kucoin spot"
            if kucoin_source and symbol != f"{coin}-USDT":
                raise ValueError(f"{coin}: source symbol does not match the requested USDT pair")
            if (binance_source or simple_usdt_pair) and symbol != f"{coin}USDT":
                raise ValueError(f"{coin}: source symbol does not match the requested USDT pair")
        frame.attrs.update({"coin": coin, "data_sha256": digest, "provenance": metadata})
        frames[coin] = frame
    return frames


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _request(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "crypto-forecast-research/0.1"})
    for attempt in range(3):
        try:
            # All coin/page/archive pools in this worker share the same bound.
            with _HTTP_SLOTS:
                with urlopen(request, timeout=30) as response:
                    return response.read()
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
        except (URLError, ssl.SSLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
        time.sleep(0.25 * 2**attempt)
    raise RuntimeError("Unreachable request retry state")


def _page_windows(ranges, window):
    for left, right in ranges:
        cursor = left
        while cursor < right:
            end = min(cursor + window, right)
            yield cursor, end
            cursor = end


def _parallel_pages(windows, fetch, on_result):
    """Bound queued work; report completions here and return chronological results."""
    iterator = enumerate(windows)
    pending, results = {}, {}
    with ThreadPoolExecutor(max_workers=NETWORK_WORKERS) as executor:
        def fill():
            while len(pending) < NETWORK_WORKERS:
                try:
                    index, (left, right) = next(iterator)
                except StopIteration:
                    break
                pending[executor.submit(fetch, left, right)] = index

        try:
            fill()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    results[index] = future.result()
                    on_result(results[index])
                fill()
        finally:
            # A failing page/callback must not leave thousands of queued requests.
            for future in pending:
                future.cancel()
    return [results[index] for index in sorted(results)]


def _archive_frame(payload: bytes, coin: str, as_of: pd.Timestamp, interval: str = "1d") -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = [entry for entry in archive.namelist() if entry.endswith(".csv")]
        if len(entries) != 1:
            raise ValueError("Expected exactly one candle CSV in archive")
        raw = pd.read_csv(io.BytesIO(archive.read(entries[0])), header=None, dtype=str)
    if raw.shape[1] != 12:
        raise ValueError("Binance spot candle archive must contain 12 fields")
    if not raw.empty and str(raw.iloc[0, 0]).lower() in {"open_time", "open time"}:
        raw = raw.iloc[1:]
    frame = raw.iloc[:, :6].copy()
    frame.columns = COLUMNS
    return _normalize(frame, coin, as_of, interval)


def _kucoin_page(coin, interval, start, end, cutoff, *, cache_dir=None, refresh=False):
    """Time-bounded public candles: KuCoin returns newest first, O,C,H,L,V."""
    query = urlencode({"symbol": f"{coin}-USDT", "type": {"15m": "15min", "1d": "1day"}[interval],
                       "startAt": int(start.timestamp()), "endAt": int(end.timestamp()) - 1})
    url = KUCOIN_CANDLES + "?" + query
    path = None if cache_dir is None else Path(cache_dir) / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    if path is not None and path.exists() and not refresh:
        cached = json.loads(path.read_bytes())
        payload = cached["payload"].encode()
        manifest = cached["manifest"]
        if manifest["url"] != url or hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
            raise ValueError(f"{coin}: KuCoin cached response checksum mismatch")
    else:
        payload = _request(url)
        manifest = {"url": url, "sha256": hashlib.sha256(payload).hexdigest(),
                    "fetched_at": _utc().isoformat(), "source": "KuCoin spot", "interval": interval}
    response = json.loads(payload)
    if not isinstance(response, dict) or response.get("code") != "200000" or not isinstance(response.get("data"), list):
        raise ValueError(f"{coin}: invalid KuCoin candle response: {str(response)[:200]}")
    rows = response["data"]
    if len(rows) > 1500 or any(not isinstance(row, list) or len(row) != 7 for row in rows):
        raise ValueError(f"{coin}: invalid KuCoin candle response schema")
    frame = pd.DataFrame([[row[0], row[1], row[3], row[4], row[2], row[5]] for row in rows], columns=COLUMNS)
    frame = _normalize(frame, coin, cutoff, interval)
    frame = frame.loc[(frame.timestamp >= start) & (frame.timestamp < end)].reset_index(drop=True)
    manifest = {**manifest, "rows": len(frame)}
    if path is not None:
        _atomic_write(path, _json_bytes({"payload": payload.decode(), "manifest": manifest}))
    return frame, manifest


def _spot_first_dates(coin, now, cache_dir, refresh=False, *, on_progress=None):
    """Discover actual first 15m candles; daily KuCoin pages locate the first day.

    KuCoin has no earliest-candle query. Scan forward from before its launch in
    bounded 1500-day pages, then inspect that day's 15m candles. Empty older
    windows are safe to cache. The final comparison is cached for one UTC day.
    """
    path = Path(cache_dir) / "exchange-comparison.json"
    _progress(on_progress, coin, "checking_source", detail="comparing earliest candles")
    if path.exists() and not refresh:
        saved = json.loads(path.read_bytes())
        if saved.get("checked_on") == now.date().isoformat() and saved.get("coin") == coin:
            _progress(on_progress, coin, "checking_source", detail="cached exchange comparison")
            return saved["exchanges"]
    comparison = {}
    for source in ("Binance spot", "KuCoin spot"):
        _progress(on_progress, coin, "checking_source", source=source, detail="checking earliest candle")
        entry = {"first_date": None}
        try:
            if source == "Binance spot":
                url = "https://data-api.binance.vision/api/v3/klines?" + urlencode(
                    {"symbol": f"{coin}USDT", "interval": "15m", "startTime": 0, "limit": 1})
                rows = json.loads(_request(url))
                if not isinstance(rows, list) or any(not isinstance(row, list) or len(row) != 12 for row in rows):
                    raise ValueError(f"{coin}: invalid Binance first-candle response")
                frame = _normalize(pd.DataFrame([row[:6] for row in rows], columns=COLUMNS), coin, now, "15m")
                if not frame.empty:
                    entry["first_date"] = frame.timestamp.min().isoformat()
                entry["url"] = url
            else:
                cursor = _utc("2017-01-01")
                while cursor < now:
                    end = min(cursor + 1500 * DAY, now.ceil("D"))
                    daily, _ = _kucoin_page(coin, "1d", cursor, end, end,
                                            cache_dir=Path(cache_dir) / "kucoin-probes", refresh=refresh)
                    for first_day in daily.timestamp:
                        probe_end = min(first_day + DAY, now.floor("15min"))
                        if first_day >= probe_end:
                            continue
                        frame, manifest = _kucoin_page(coin, "15m", first_day, probe_end, now,
                                                      cache_dir=Path(cache_dir) / "kucoin-probes", refresh=refresh)
                        if not frame.empty:
                            entry.update(first_date=frame.timestamp.min().isoformat(), url=manifest["url"])
                            break
                    if entry["first_date"]:
                        break
                    cursor = end
            entry["status"] = "available" if entry["first_date"] else "unavailable"
        except (OSError, URLError, ValueError, KeyError, TypeError) as error:
            entry.update(status="error", error=str(error))
        comparison[source] = entry
        _progress(on_progress, coin, "checking_source", source=source, detail=entry["status"])
    if all(entry["status"] != "error" for entry in comparison.values()):
        _atomic_write(path, _json_bytes({"coin": coin, "checked_on": now.date().isoformat(), "exchanges": comparison}))
    return comparison


def _kucoin_history(coin, start, end, now, interval, cache_dir, refresh=False, *, on_progress=None):
    window = 1500 * INTERVALS[interval]
    total = max(0, int((end - start + window - pd.Timedelta(1, unit="ns")) // window))
    completed = 0
    _progress(on_progress, coin, "history_pages", 0, total, source="KuCoin spot")
    def fetch(left, right):
        try:
            frame, manifest = _kucoin_page(coin, interval, left, right, now,
                                           cache_dir=cache_dir, refresh=refresh)
            return frame, manifest, None
        except (OSError, URLError, ValueError, KeyError, TypeError) as error:
            return None, None, f"{left.isoformat()} to {right.isoformat()}: {error}"

    def checked(result):
        nonlocal completed
        completed += 1
        _progress(on_progress, coin, "history_pages", completed, total, source="KuCoin spot",
                  detail="error" if result[2] else "checked")

    results = _parallel_pages(_page_windows([(start, end)], window), fetch, checked)
    pieces = [frame for frame, _, error in results if error is None]
    responses = [manifest for _, manifest, error in results if error is None]
    errors = [error for _, _, error in results if error is not None]
    frame = _normalize(pd.concat(pieces, ignore_index=True), coin, now, interval) if pieces else pd.DataFrame(columns=COLUMNS)
    return {"frame": frame, "archives": [], "missing_archives": [], "errors": errors, "historical_rest": responses}


def _binance_history(coin, start, end, now, interval, cache_dir, refresh, executor, *, on_progress=None):
    month_start = start.replace(day=1).floor("D")
    months = pd.date_range(month_start, end, freq="MS", inclusive="left")
    historical = [month for month in months if month < now.replace(day=1).floor("D")]
    fetch = lambda url: _fetch_archive(url, cache_dir, coin, refresh, now, interval)

    def archives(urls, phase):
        _progress(on_progress, coin, phase, 0, len(urls), source="Binance spot")
        # Consume completed work immediately while retaining submission order in
        # artifacts, duplicate resolution, errors and source manifests.
        ordered = [None] * len(urls)
        futures = {executor.submit(fetch, url): index for index, url in enumerate(urls)}
        for completed, future in enumerate(as_completed(futures), 1):
            item = future.result()
            ordered[futures[future]] = item
            detail = "cached" if item.get("manifest", {}).get("cached") else item["status"]
            _progress(on_progress, coin, phase, completed, len(urls), source="Binance spot", detail=detail)
        return ordered

    results = archives([_archive_url(coin, month, "monthly", interval) for month in historical], "monthly_archives")
    monthly = {item["url"]: item for item in results}
    daily_dates = []
    for month in months:
        result = monthly.get(_archive_url(coin, month, "monthly", interval))
        recent_missing = (result is not None and result["status"] == "missing" and month >= now.floor("D") - 45 * DAY)
        if result is None or recent_missing:
            daily_dates.extend(pd.date_range(max(start.floor("D"), month), min(end, month + pd.offsets.MonthBegin(1)),
                                             freq="D", inclusive="left"))
    results.extend(archives([_archive_url(coin, day, "daily", interval) for day in daily_dates], "daily_archives"))
    successful = [item for item in results if item["status"] == "ok"]
    pieces = [item["frame"] for item in successful]
    frame = _normalize(pd.concat(pieces, ignore_index=True), coin, now, interval) if pieces else pd.DataFrame(columns=COLUMNS)
    frame = frame.loc[(frame.timestamp >= start) & (frame.timestamp < end)].reset_index(drop=True)
    return {"frame": frame, "archives": [item["manifest"] for item in successful],
            "missing_archives": sorted(item["url"] for item in results if item["status"] == "missing"),
            "errors": [item["error"] for item in results if item["status"] == "error"], "historical_rest": []}


def _select_spot_history(coin, start, end, now, interval, cache_root, refresh, executor, *, on_progress=None):
    progress_options = {"on_progress": on_progress} if on_progress is not None else {}
    comparison = _spot_first_dates(coin, now, Path(cache_root) / coin, refresh, **progress_options)
    # Binance remains the default on equal or unknown starts. Unknown sources
    # are still attempted if the preferred download is incomplete or unusable.
    order = sorted(comparison, key=lambda source: (
        _utc(comparison[source]["first_date"]) if comparison[source]["first_date"] else _utc("2100-01-01"),
        source != "Binance spot"))
    candidates = {}
    for source in order:
        first = comparison[source]["first_date"]
        if source == "KuCoin spot" and (comparison[source]["status"] == "unavailable" or (first and _utc(first) >= end)):
            continue
        if source == "Binance spot":
            cache = Path(cache_root) / coin / "15m" if interval == "15m" else Path(cache_root) / coin
            left = max(start, _utc(first).floor(INTERVALS[interval])) if first else start
            if left >= end:
                continue
            candidate = _binance_history(coin, left, end, now, interval, cache, refresh, executor, **progress_options)
        else:
            left = max(start, _utc(first).floor(INTERVALS[interval])) if first else start
            candidate = _kucoin_history(coin, left, end, now, interval, Path(cache_root) / coin / "kucoin" / interval, refresh, **progress_options)
        _progress(on_progress, coin, "verifying", source=source, detail="checking candle coverage")
        frame = candidate["frame"]
        coverage = _interval_quality(frame, interval, start, end)
        comparison[source].update(download_rows=len(frame), download_missing_count=coverage["missing_count"],
                                  download_first_date=frame.timestamp.min().isoformat() if not frame.empty else None,
                                  download_errors=candidate["errors"])
        candidates[source] = candidate
        available_start = max(start, _utc(first).ceil(INTERVALS[interval])) if first else start
        available_missing = _interval_quality(frame, interval, available_start, end)["missing_count"]
        comparison[source]["post_listing_missing_count"] = available_missing
        _progress(on_progress, coin, "verifying", source=source,
                  detail="complete" if not available_missing and not candidate["errors"] else "incomplete; comparing alternative source")
        if not frame.empty and not available_missing and not candidate["errors"]:
            break
    usable = [source for source, candidate in candidates.items() if not candidate["frame"].empty]
    def score(source):
        entry = comparison[source]
        return (bool(entry["download_errors"]), bool(entry["post_listing_missing_count"]),
                _utc(entry["download_first_date"]), entry["download_missing_count"],
                _utc(entry["first_date"] or entry["download_first_date"]), source != "Binance spot")
    source = min(usable, key=score) if usable else next(iter(candidates), "Binance spot")
    chosen = candidates.get(source, {"frame": pd.DataFrame(columns=COLUMNS), "archives": [],
                                     "errors": [], "missing_archives": [], "historical_rest": []})
    _progress(on_progress, coin, "verifying", source=source, detail="selected source")
    return {**chosen, "source": source, "symbol": f"{coin}-USDT" if source == "KuCoin spot" else f"{coin}USDT",
            "exchange_comparison": comparison,
            "selection_rule": "prefer oldest history; on incomplete downloads prefer usable post-listing coverage; Binance on ties"}


def _fetch_archive(url: str, cache_dir: Path, coin: str, refresh: bool, as_of: pd.Timestamp, interval: str = "1d"):
    filename = url.rsplit("/", 1)[-1]
    cache = cache_dir / filename
    checksum_file = cache.with_name(cache.name + ".CHECKSUM")
    manifest_file = cache.with_name(cache.name + ".metadata.json")
    try:
        cached = False
        if not refresh and cache.exists() and checksum_file.exists() and manifest_file.exists():
            payload = cache.read_bytes()
            checksum = checksum_file.read_bytes()
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if manifest.get("url") != url or manifest.get("interval") != interval:
                raise ValueError(f"Cached archive provenance mismatch for {filename}; use refresh")
            cached = True
        else:
            payload = _request(url)
            checksum = _request(url + ".CHECKSUM")
            manifest = {"url": url, "fetched_at": _utc().isoformat(), "source": "Binance spot", "interval": interval}
        match = re.search(r"\b[0-9a-fA-F]{64}\b", checksum.decode("ascii"))
        digest = hashlib.sha256(payload).hexdigest()
        if match is None or match.group().lower() != digest:
            raise ValueError(f"SHA256 checksum mismatch for {filename}")
        if cached and manifest.get("sha256") != digest:
            raise ValueError(f"Cached manifest checksum mismatch for {filename}; use refresh")
        frame = _archive_frame(payload, coin, as_of, interval)
        manifest.update({"sha256": digest, "official_checksum_verified": True,
                         "cache_file": str(cache.resolve()), "cached": cached})
        if not cached:
            _atomic_write(cache, payload)
            _atomic_write(checksum_file, checksum)
            _atomic_write(manifest_file, _json_bytes(manifest))
        return {"status": "ok", "frame": frame, "manifest": manifest, "url": url}
    except HTTPError as error:
        if error.code == 404 and error.geturl() == url:
            return {"status": "missing", "url": url}
        return {"status": "error", "url": url, "error": f"HTTP {error.code}: {error.geturl()} (archive/checksum download)"}
    except Exception as error:
        return {"status": "error", "url": url, "error": f"{url}: {type(error).__name__}: {error}"}


def _archive_url(coin: str, period: pd.Timestamp, frequency: str, interval: str = "1d") -> str:
    interval_delta(interval)
    symbol = f"{coin}USDT"
    date = period.strftime("%Y-%m" if frequency == "monthly" else "%Y-%m-%d")
    return f"{ARCHIVE_BASE}/{frequency}/klines/{symbol}/{interval}/{symbol}-{interval}-{date}.zip"


def download_history(coins: list[str], start, end, data_dir: Path | str, refresh=False) -> dict:
    """Choose Binance/KuCoin for [start, end) UTC daily history with caching.

    Returns metadata keyed by coin, including partial/failed status and missing
    archive URLs. HTTP 404 (often a pre-listing period) is explicit. A missing
    recent monthly file triggers daily fallback; older absence remains reported
    without requesting each pre-listing day. Other monthly errors remain errors.
    Existing CSV/provenance files survive a fetch with no usable rows.
    """
    selected = _coins(coins)
    start, requested_end, now = _utc(start), _utc(end), _utc()
    if start != start.floor("D") or requested_end != requested_end.floor("D"):
        raise ValueError("Download start/end must be UTC midnight dates")
    if start >= requested_end:
        raise ValueError("Download start must precede exclusive end")
    end = min(requested_end, now.floor("D"))
    if start >= end:
        raise ValueError("Requested range contains no completed days")
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for coin in selected:
            acquired = _select_spot_history(coin, start, end, now, "1d", directory / "_archives", refresh, executor)
            frame = acquired["frame"]
            unavailable = _missing_dates(frame, start, end)
            errors = acquired["errors"]
            metadata = {"coin": coin, "symbol": acquired["symbol"], "source": acquired["source"],
                        "exchange_comparison": acquired["exchange_comparison"], "selection_rule": acquired["selection_rule"],
                        "historical_rest": acquired["historical_rest"], "quote_currency": "USDT", "interval": "1d", "timestamp_semantics": "UTC candle open",
                        "requested_start": start.isoformat(), "requested_end_exclusive": requested_end.isoformat(),
                        "effective_end_exclusive": end.isoformat(), "fetched_at": now.isoformat(),
                        "rows": len(frame), "unavailable_dates": unavailable,
                        "missing_dates": _missing_dates(frame),
                        "missing_archives": acquired["missing_archives"],
                        "errors": errors, "archives": acquired["archives"],
                        "status": "failed" if frame.empty else ("partial" if unavailable or errors else "downloaded")}
            if not frame.empty:
                csv_payload = frame[COLUMNS].to_csv(index=False).encode()
                metadata["data_sha256"] = hashlib.sha256(csv_payload).hexdigest()
                metadata["first_date"] = frame.timestamp.min().isoformat()
                metadata["last_date"] = frame.timestamp.max().isoformat()
                _atomic_write(directory / f"{coin}.csv", csv_payload)
                _atomic_write(directory / f"{coin}.metadata.json", _json_bytes(metadata))
            report[coin] = metadata
    return report


def _merge_ranges(ranges):
    merged = []
    for left, right in sorted((_utc(left), _utc(right)) for left, right in ranges):
        if left >= right:
            continue
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    return merged


def _kucoin_no_trade_ranges(metadata):
    """Successful bounded KuCoin pages omit no-tick slots, not failed downloads.

    Read coverage from retained nonempty response manifests, including legacy
    caches. Wholly empty pages remain retryable; other sources and unbounded
    or wrong-instrument requests confer no coverage.
    """
    if metadata.get("source") != "KuCoin spot":
        return []
    ranges = []
    for manifest in [*metadata.get("historical_rest", []), *metadata.get("recent_rest", [])]:
        url = urlparse(manifest.get("url", ""))
        query = parse_qs(url.query)
        if (url.scheme != "https" or url.netloc != "api.kucoin.com" or url.path != "/api/v1/market/candles"
                or manifest.get("source") != "KuCoin spot" or manifest.get("interval") != "15m"
                or not manifest.get("rows")
                or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("sha256", ""))
                or query.get("type") != ["15min"] or query.get("symbol") != [metadata.get("symbol")]):
            continue
        try:
            left = pd.Timestamp(int(query["startAt"][0]), unit="s", tz="UTC")
            right = pd.Timestamp(int(query["endAt"][0]) + 1, unit="s", tz="UTC")
        except (KeyError, ValueError, IndexError, OverflowError):
            continue
        delta = INTERVALS["15m"]
        if left == left.floor(delta) and right == right.floor(delta) and pd.Timedelta(0) < right - left <= 1500 * delta:
            ranges.append((left, right))
    return _merge_ranges(ranges)


def aggregate_candles(frame: pd.DataFrame, interval: str, as_of=None, coin: str = "candles", *, no_trade_ranges=()) -> pd.DataFrame:
    """Build complete UTC buckets from real 15-minute candles, summing volume.

    Missing bars remove the larger bucket unless a fully checked exchange
    response range explicitly confirms that absent slots mean no trades.
    Such sparse buckets use only observed OHLCV; no prices or rows are filled.
    """
    cutoff = _utc(as_of)
    delta = interval_delta(interval)
    base = _normalize(frame, coin, cutoff, "15m")
    if interval == "15m" or base.empty:
        result = base.copy()
        result.attrs["quality"] = _interval_quality(result, interval)
        return result
    indexed = base.set_index("timestamp")
    grouped = indexed.resample(delta, origin="start_day", label="left", closed="left")
    aggregated = grouped.agg({"open": "first", "high": "max", "low": "min",
                              "close": "last", "volume": "sum"})
    count = grouped["close"].count()
    full_count = count == int(delta / INTERVALS["15m"])
    covered = np.zeros(len(aggregated), dtype=bool)
    for left, right in _merge_ranges(no_trade_ranges):
        covered |= (aggregated.index >= left) & (aggregated.index + delta <= right)
    complete = (full_count | (covered & (count > 0))) & (aggregated.index + delta <= cutoff)
    result = aggregated.loc[complete].reset_index()[COLUMNS]
    result = _normalize(result, coin, cutoff, interval)
    result.attrs["quality"]["incomplete_buckets_dropped"] = int((~complete).sum())
    result.attrs["quality"]["sparse_buckets_included"] = int((complete & ~full_count).sum())
    return result


def _multiframe_provenance(path: Path, coin: str, interval: str, digest: str) -> dict:
    sidecar = path.with_suffix(".metadata.json")
    if not sidecar.exists():
        raise ValueError(f"{coin} {interval}: provenance sidecar required: {sidecar}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{coin} {interval}: provenance must be a JSON object")
    if not str(metadata.get("source", "")).strip():
        raise ValueError(f"{coin} {interval}: provenance must identify its source")
    if str(metadata.get("coin", "")).strip().upper() != coin:
        raise ValueError(f"{coin} {interval}: provenance identifies a different or missing coin")
    if metadata.get("interval") != interval:
        raise ValueError(f"{coin} {interval}: source interval must be {interval}")
    quotes = [str(metadata[field]).strip().upper() for field in ("quote_currency", "quote") if field in metadata]
    if not quotes or any(quote != "USDT" for quote in quotes):
        raise ValueError(f"{coin} {interval}: source quote currency must be USDT")
    if metadata.get("data_sha256") and metadata["data_sha256"] != digest:
        raise ValueError(f"{coin} {interval}: CSV hash differs from its provenance metadata")
    if "symbol" in metadata:
        symbol = str(metadata["symbol"]).strip().upper()
        binance_source = str(metadata["source"]).strip().casefold() == "binance spot"
        kucoin_source = str(metadata["source"]).strip().casefold() == "kucoin spot"
        if kucoin_source and symbol != f"{coin}-USDT":
            raise ValueError(f"{coin} {interval}: source symbol does not match the requested USDT pair")
        if (binance_source or re.fullmatch(r"[A-Z0-9]+USDT", symbol)) and symbol != f"{coin}USDT":
            raise ValueError(f"{coin} {interval}: source symbol does not match the requested USDT pair")
    if "timestamp_semantics" in metadata and metadata["timestamp_semantics"] != "UTC candle open":
        raise ValueError(f"{coin} {interval}: timestamps must identify UTC candle opens")
    return metadata


def load_multiframe_data(data_dir: Path | str, coins: list[str], as_of=None) -> dict[str, dict[str, pd.DataFrame]]:
    """Load all four real OHLCV histories with explicit, consistent provenance.

    Files are ``INTERVAL/COIN.csv`` and ``INTERVAL/COIN.metadata.json``. Every
    source must identify coin, source, interval, and USDT quote. An ``as_of``
    cutoff can be intraday; only fully closed candles survive it.
    """
    directory, cutoff = Path(data_dir), _utc(as_of)
    histories = {}
    for coin in _coins(coins):
        frames, sources, symbols = {}, set(), set()
        for interval in TIMEFRAMES:
            path = directory / interval / f"{coin}.csv"
            if not path.exists():
                raise ValueError(f"{coin}: missing {interval} history at {path}; download native 15m history first")
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            metadata = _multiframe_provenance(path, coin, interval, digest)
            frame = _normalize(pd.read_csv(io.BytesIO(payload)), coin, cutoff, interval)
            if frame.empty:
                raise ValueError(f"{coin}: no completed {interval} candles available")
            sources.add(str(metadata["source"]).strip().casefold())
            if "symbol" in metadata:
                symbols.add(str(metadata["symbol"]).strip().upper())
            frame.attrs.update({"coin": coin, "interval": interval, "data_sha256": digest,
                                "provenance": metadata})
            frames[interval] = frame
        if len(sources) != 1 or len(symbols) > 1:
            raise ValueError(f"{coin}: all timeframes must use the same source and instrument")
        for interval, frame in frames.items():
            aggregation = frame.attrs["provenance"].get("aggregation")
            if aggregation is not None:
                if not isinstance(aggregation, dict) or aggregation.get("source_interval") != "15m":
                    raise ValueError(f"{coin} {interval}: aggregation provenance must identify its 15m source")
                if aggregation.get("source_data_sha256") != frames["15m"].attrs["data_sha256"]:
                    raise ValueError(f"{coin} {interval}: aggregated candles refer to a different 15m source file")
        histories[coin] = frames
    return histories


def _incremental_spot_history(coin, start, end, now, directory, cache_root, on_progress):
    """Reuse verified native candles and request only absent slots on their source."""
    path = Path(directory) / "15m" / f"{coin}.csv"
    if not path.exists():
        return None
    _progress(on_progress, coin, "checking_local", detail="verifying saved candle coverage")
    payload = path.read_bytes()
    original = _multiframe_provenance(path, coin, "15m", hashlib.sha256(payload).hexdigest())
    base = _normalize(pd.read_csv(io.BytesIO(payload)), coin, now, "15m")
    if base.empty:
        return None
    source = original["source"]
    if source not in {"Binance spot", "KuCoin spot"}:
        raise ValueError(f"{coin}: incremental download requires Binance spot or KuCoin spot provenance")
    comparison = original.get("exchange_comparison", {})
    first = comparison.get(source, {}).get("first_date")
    if first is None and start < base.timestamp.iloc[0]:
        comparison = _spot_first_dates(coin, now, Path(cache_root) / coin, on_progress=on_progress)
        first = comparison.get(source, {}).get("first_date")
    delta = INTERVALS["15m"]
    left = max(start, _utc(first).ceil(delta)) if first else start
    actual = pd.DatetimeIndex(base.loc[(base.timestamp >= left) & (base.timestamp < end), "timestamp"]).as_unit("ns")
    edges = np.concatenate(([left.value - delta.value], actual.asi8, [end.value]))
    ranges = [(pd.Timestamp(edges[index] + delta.value, tz="UTC"), pd.Timestamp(edges[index + 1], tz="UTC"))
              for index in np.flatnonzero(np.diff(edges) > delta.value)] if left < end else []
    # Successful historical KuCoin requests already checked omitted no-tick
    # slots. Re-query only ranges that have no such response coverage.
    checked_ranges = _kucoin_no_trade_ranges(original)
    if checked_ranges:
        unchecked = []
        for gap_left, gap_right in ranges:
            cursor = gap_left
            for checked_left, checked_right in checked_ranges:
                if checked_right <= cursor:
                    continue
                if checked_left >= gap_right:
                    break
                if cursor < checked_left:
                    unchecked.append((cursor, min(checked_left, gap_right)))
                cursor = max(cursor, checked_right)
                if cursor >= gap_right:
                    break
            if cursor < gap_right:
                unchecked.append((cursor, gap_right))
        ranges = unchecked
    page_slots = 1500 if source == "KuCoin spot" else 1000
    window = page_slots * delta
    total = sum(int((right - left + window - pd.Timedelta(1, unit="ns")) // window) for left, right in ranges)
    completed = 0
    _progress(on_progress, coin, "history_pages", 0, total, source=source,
              detail="fetching missing candles" if ranges else "requested candles already saved")
    def fetch(cursor, page_end):
        if source == "KuCoin spot":
            frame, manifest = _kucoin_page(coin, "15m", cursor, page_end, now,
                                              cache_dir=Path(cache_root) / coin / "kucoin" / "15m", refresh=True)
        else:
            query = urlencode({"symbol": f"{coin}USDT", "interval": "15m", "limit": 1000,
                                   "startTime": int(cursor.timestamp() * 1000),
                                   "endTime": int(page_end.timestamp() * 1000) - 1})
            url = "https://data-api.binance.vision/api/v3/klines?" + query
            response = _request(url)
            rows = json.loads(response)
            if (not isinstance(rows, list) or len(rows) > 1000 or
                        any(not isinstance(row, list) or len(row) != 12 for row in rows)):
                raise ValueError(f"{coin}: invalid public candle response schema")
            frame = _normalize(pd.DataFrame([row[:6] for row in rows], columns=COLUMNS), coin, now, "15m")
            manifest = {"url": url, "sha256": hashlib.sha256(response).hexdigest(),
                            "fetched_at": now.isoformat(), "rows": len(frame)}
        frame = frame.loc[(frame.timestamp >= cursor) & (frame.timestamp < page_end)]
        return frame, manifest

    def checked(result):
        nonlocal completed
        completed += 1
        _progress(on_progress, coin, "history_pages", completed, total, source=source, detail="checked missing candles")

    results = _parallel_pages(_page_windows(ranges, window), fetch, checked)
    pieces = [frame for frame, _ in results]
    responses = [manifest for _, manifest in results]
    reused_rows = len(base)
    if pieces:
        base = _normalize(pd.concat([base, *pieces], ignore_index=True), coin, now, "15m")
    return {"frame": base, "source": source, "symbol": original["symbol"],
            "exchange_comparison": comparison, "selection_rule": original.get("selection_rule", "reuse saved exchange"),
            "archives": original.get("archives", []), "missing_archives": [], "errors": [],
            "historical_rest": [*original.get("historical_rest", []), *original.get("recent_rest", []), *responses],
            "reused_rows": reused_rows, "fetched_rows": len(base) - reused_rows,
            "coverage_start": min(start, _utc(original.get("requested_start", base.timestamp.iloc[0]))),
            "coverage_end": max(end, base.timestamp.iloc[-1] + delta)}


def download_multiframe_history(coins: list[str], start, end, data_dir: Path | str, refresh=False, *, archive_root=None, on_progress=None, existing_data_dir=None) -> dict:
    """Choose older usable spot history and derive complete 1h, 4h and daily OHLCV.

    Binance archives retain official checksums; KuCoin responses retain hashes
    and source manifests. Both exchanges' actual first 15m candles are compared.
    Incomplete preferred history triggers comparison with the other source.
    Only four network
    workers run at once. The original root-level daily CSVs are left intact.
    ``start`` and exclusive ``end`` must align to the UTC 15-minute grid.
    ``archive_root`` allows staged downloads to reuse the verified archive cache.
    ``existing_data_dir`` reuses verified native CSV candles, fetching only gaps
    on the saved exchange and retaining candles outside the requested range.
    ``on_progress`` receives per-coin phase dictionaries with checked work
    counts, or unknown counts for source checks/verification. Callback failures
    propagate. Archive events run on this caller's thread, in completion order.
    """
    selected = _coins(coins)
    start, requested_end, now = _utc(start), _utc(end), _utc()
    base_delta = INTERVALS["15m"]
    if start != start.floor(base_delta) or requested_end != requested_end.floor(base_delta):
        raise ValueError("Download start/end must align to the UTC 15m interval grid")
    if start >= requested_end:
        raise ValueError("Download start must precede exclusive end")
    end = min(requested_end, now.floor(base_delta))
    if start >= end:
        raise ValueError("Requested range contains no completed 15m candles")
    directory = Path(data_dir)
    report = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for coin in selected:
            cache_root = Path(archive_root) if archive_root is not None else directory / "_archives"
            progress_options = {"on_progress": on_progress} if on_progress is not None else {}
            acquired = (_incremental_spot_history(coin, start, end, now, existing_data_dir, cache_root, on_progress)
                        if existing_data_dir is not None and not refresh else None)
            if acquired is None:
                acquired = _select_spot_history(coin, start, end, now, "15m", cache_root, refresh, executor, **progress_options)
            base = acquired["frame"]
            errors = acquired["errors"]
            missing_archives = acquired["missing_archives"]
            common = {"coin": coin, "symbol": acquired["symbol"], "source": acquired["source"],
                      "exchange_comparison": acquired["exchange_comparison"], "selection_rule": acquired["selection_rule"],
                      "quote_currency": "USDT", "timestamp_semantics": "UTC candle open",
                      "volume_semantics": "base-asset traded volume",
                      "requested_start": acquired.get("coverage_start", start).isoformat(), "requested_end_exclusive": requested_end.isoformat(),
                      "effective_end_exclusive": acquired.get("coverage_end", end).isoformat(), "fetched_at": now.isoformat(),
                      "missing_archives": missing_archives, "errors": errors}
            entries = {}
            base_digest = None
            source_archives = [{"url": item["url"], "sha256": item["sha256"]} for item in acquired["archives"]]
            _progress(on_progress, coin, "timeframes", 0, len(TIMEFRAMES), source=acquired["source"])
            no_trade_ranges = _kucoin_no_trade_ranges(acquired)
            frames = {interval: base if interval == "15m" else aggregate_candles(base, interval, now, coin, no_trade_ranges=no_trade_ranges)
                      for interval in TIMEFRAMES}
            publish = all(not frame.empty for frame in frames.values())
            payloads = {}
            serialized = 0
            for interval in TIMEFRAMES:
                frame = frames[interval]
                delta = INTERVALS[interval]
                quality = _interval_quality(frame, interval)
                quality.update({key: value for key, value in frame.attrs.get("quality", {}).items()
                                if key in {"incomplete_buckets_dropped", "sparse_buckets_included"}})
                coverage = _interval_quality(frame, interval, acquired.get("coverage_start", start).ceil(delta),
                                             acquired.get("coverage_end", end).floor(delta))
                metadata = {**common, "interval": interval, "rows": len(frame), "quality": quality,
                            "persisted": publish,
                            "unavailable_count": coverage["missing_count"],
                            "unavailable_dates": coverage["missing_dates"],
                            "unavailable_dates_truncated": coverage["missing_dates_truncated"],
                            "status": "failed" if frame.empty else ("partial" if coverage["missing_count"] or errors else "downloaded")}
                if interval == "15m":
                    metadata["archives"] = acquired["archives"]
                    metadata["historical_rest"] = acquired["historical_rest"]
                else:
                    metadata["aggregation"] = {
                        "source_interval": "15m", "source_data_sha256": base_digest,
                        "source_archives": source_archives,
                        "formula": "open=first; high=max; low=min; close=last; volume=sum",
                        "buckets": "UTC-aligned, left-closed, labeled by candle open",
                        "completeness": ("all slots required outside fully checked KuCoin no-tick response ranges; no filling"
                                         if no_trade_ranges else f"all {int(delta / base_delta)} unique 15m candles required; no filling")}
                    if no_trade_ranges:
                        metadata["aggregation"]["no_trade_policy"] = "KuCoin omits no-tick intervals; aggregate observed OHLCV only in fully checked response ranges"
                if not frame.empty:
                    payload = frame[COLUMNS].to_csv(index=False).encode()
                    digest = hashlib.sha256(payload).hexdigest()
                    metadata.update(data_sha256=digest, first_date=frame.timestamp.min().isoformat(),
                                    last_date=frame.timestamp.max().isoformat())
                    if interval == "15m":
                        base_digest = digest
                    payloads[interval] = (payload, _json_bytes(metadata))
                    serialized += 1
                    _progress(on_progress, coin, "timeframes", serialized, len(TIMEFRAMES),
                              source=acquired["source"], detail=interval)
                entries[interval] = metadata
            # An empty derived interval must not leave another exchange's old
            # files beside newly downloaded data. Validate/serialize the whole
            # set before replacing any of the previous candle files.
            if publish:
                for interval, (payload, sidecar) in payloads.items():
                    _atomic_write(directory / interval / f"{coin}.csv", payload)
                    _atomic_write(directory / interval / f"{coin}.metadata.json", sidecar)
            states = [entry["status"] for entry in entries.values()]
            report[coin] = {**common, "rows": len(base), "timeframes": entries, "persisted": publish,
                            "status": "failed" if not publish else ("downloaded" if all(state == "downloaded" for state in states) else "partial")}
            if "reused_rows" in acquired:
                report[coin].update(reused_rows=acquired["reused_rows"], fetched_rows=acquired["fetched_rows"])
    return report


def refresh_multiframe_recent(data_dir: Path | str, coins: list[str], as_of=None, max_days=7, *, on_progress=None) -> dict:
    """Append closed public 15m candles and rebuild complete larger buckets.

    This updates local data on demand. It never schedules forecasts or orders.
    Binance rows must continue without missing slots. Fully checked KuCoin
    pages may omit no-tick slots; those slots stay absent from native history.
    ``on_progress`` reports checked recent pages and serialized timeframes.
    Binance page totals remain unknown until pagination finishes, since public
    responses may return fewer than the maximum 1000 candles.
    """
    cutoff = min(_utc(as_of), _utc()).floor(INTERVALS["15m"])
    directory = Path(data_dir)
    if max_days <= 0 or not np.isfinite(max_days):
        raise ValueError("max_days must be finite and positive")
    result = {}
    for coin in _coins(coins):
        _progress(on_progress, coin, "checking_source", detail="loading local provenance")
        frames = load_multiframe_data(directory, [coin], as_of=cutoff)[coin]
        base = frames["15m"].copy()
        original = dict(base.attrs["provenance"])
        source = original.get("source")
        if source not in {"Binance spot", "KuCoin spot"}:
            raise ValueError(f"{coin}: public refresh requires Binance spot or KuCoin spot source history")
        first_new = base.timestamp.iloc[-1] + INTERVALS["15m"]
        if source == "KuCoin spot":
            for left, right in _kucoin_no_trade_ranges(original):
                if left <= first_new <= right:
                    first_new = max(first_new, min(right, cutoff))
        if cutoff - first_new > pd.Timedelta(days=max_days):
            raise ValueError(f"{coin}: archive history is over {max_days} days behind; download historical archives first")
        cursor = first_new
        pieces, responses = [], []
        page_window = 1500 * INTERVALS["15m"]
        total_pages = (max(0, int((cutoff - first_new + page_window - pd.Timedelta(1, unit="ns")) // page_window))
                       if source == "KuCoin spot" else (0 if cursor >= cutoff else None))
        completed_pages = 0
        _progress(on_progress, coin, "recent_pages", 0, total_pages, source=source)
        while cursor < cutoff:
            if source == "KuCoin spot":
                page_end = min(cursor + 1500 * INTERVALS["15m"], cutoff)
                try:
                    frame, manifest = _kucoin_page(coin, "15m", cursor, page_end, cutoff)
                finally:
                    completed_pages += 1
                    _progress(on_progress, coin, "recent_pages", completed_pages, total_pages, source=source)
            else:
                query = urlencode({"symbol": f"{coin}USDT", "interval": "15m", "limit": 1000,
                                   "startTime": int(cursor.timestamp()*1000),
                                   "endTime": int(cutoff.timestamp()*1000)-1})
                url = "https://data-api.binance.vision/api/v3/klines?" + query
                try:
                    payload = _request(url)
                finally:
                    completed_pages += 1
                    _progress(on_progress, coin, "recent_pages", completed_pages, total_pages, source=source)
                rows = json.loads(payload)
                if not isinstance(rows, list) or not rows:
                    raise ValueError(f"{coin}: public candles did not cover the requested recent interval")
                if any(not isinstance(row, list) or len(row) != 12 for row in rows):
                    raise ValueError(f"{coin}: invalid public candle response schema")
                frame = _normalize(pd.DataFrame([row[:6] for row in rows], columns=COLUMNS), coin, cutoff, "15m")
                manifest = {"url": url, "sha256": hashlib.sha256(payload).hexdigest(),
                            "fetched_at": _utc().isoformat(), "rows": len(frame)}
            frame = frame.loc[(frame.timestamp >= cursor) & (frame.timestamp < cutoff)]
            if source == "KuCoin spot":
                pieces.append(frame)
                responses.append(manifest)
                cursor = page_end
                continue
            if frame.empty or frame.timestamp.iloc[0] != cursor:
                raise ValueError(f"{coin}: gap in recent public candle response at {cursor}")
            quality = _interval_quality(frame, "15m")
            if quality["missing_count"]:
                raise ValueError(f"{coin}: recent public candles contain missing 15m intervals")
            pieces.append(frame)
            responses.append(manifest)
            cursor = frame.timestamp.iloc[-1] + INTERVALS["15m"]
        if total_pages is None:
            _progress(on_progress, coin, "recent_pages", completed_pages, completed_pages, source=source)
        if not pieces:
            result[coin] = {"status": "current", "added_rows": 0, "as_of": first_new.isoformat()}
            continue
        base = _normalize(pd.concat([base, *pieces], ignore_index=True), coin, cutoff, "15m")
        _progress(on_progress, coin, "timeframes", 0, len(TIMEFRAMES), source=source)
        payloads = {}
        base_hash = None
        no_trade_ranges = _kucoin_no_trade_ranges({**original, "recent_rest": [*original.get("recent_rest", []), *responses]})
        for completed, interval in enumerate(TIMEFRAMES, 1):
            frame = base if interval == "15m" else aggregate_candles(base, interval, cutoff, coin, no_trade_ranges=no_trade_ranges)
            csv_payload = frame[COLUMNS].to_csv(index=False).encode()
            digest = hashlib.sha256(csv_payload).hexdigest()
            delta = INTERVALS[interval]
            coverage = _interval_quality(frame, interval,
                                         _utc(original.get("requested_start", base.timestamp.min())).ceil(delta),
                                         cutoff.floor(delta))
            metadata = {**original, "interval": interval, "rows": len(frame),
                        "data_sha256": digest, "quality": _interval_quality(frame, interval),
                        "unavailable_count": coverage["missing_count"],
                        "unavailable_dates": coverage["missing_dates"],
                        "unavailable_dates_truncated": coverage["missing_dates_truncated"],
                        "status": "partial" if coverage["missing_count"] or original.get("errors") else "downloaded",
                        "effective_end_exclusive": cutoff.isoformat(), "fetched_at": _utc().isoformat(),
                        "first_date": frame.timestamp.min().isoformat(),
                        "last_date": frame.timestamp.max().isoformat(),
                        "recent_rest": [*original.get("recent_rest", []), *responses]}
            if interval == "15m":
                base_hash = digest
                metadata.pop("aggregation", None)
            else:
                metadata.pop("archives", None)
                metadata["aggregation"] = {"source_interval": "15m", "source_data_sha256": base_hash,
                                           "formula": "open=first; high=max; low=min; close=last; volume=sum",
                                           "completeness": ("all slots required outside fully checked KuCoin no-tick response ranges; no filling"
                                                            if no_trade_ranges else "all unique 15m slots required; no filling")}
                if no_trade_ranges:
                    metadata["aggregation"]["no_trade_policy"] = "KuCoin omits no-tick intervals; aggregate observed OHLCV only in fully checked response ranges"
            payloads[interval] = (csv_payload, _json_bytes(metadata))
            _progress(on_progress, coin, "timeframes", completed, len(TIMEFRAMES), source=source, detail=interval)
        # Prepare all bytes first so bad network/schema responses cannot replace files.
        for interval, (csv_payload, metadata_payload) in payloads.items():
            _atomic_write(directory / interval / f"{coin}.csv", csv_payload)
            _atomic_write(directory / interval / f"{coin}.metadata.json", metadata_payload)
        result[coin] = {"status": "refreshed", "added_rows": sum(len(piece) for piece in pieces),
                        "as_of": cutoff.isoformat(), "responses": responses}
    return result
