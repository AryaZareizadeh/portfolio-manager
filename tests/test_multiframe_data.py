"""Multi-interval data contracts: real aggregation, fake only network bytes."""

import hashlib
import io
import json
import ssl
from urllib.error import HTTPError, URLError
import zipfile

import numpy as np
import pandas as pd
import pytest

from crypto_forecast import data
from crypto_forecast.data import aggregate_candles, download_multiframe_history, load_multiframe_data
from crypto_forecast.timeframes import INTERVALS, TIMEFRAMES


def base_candles(periods=192, start="2025-01-01"):
    timestamps = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    prices = np.arange(periods, dtype=float) + 100
    return pd.DataFrame({"timestamp": timestamps, "open": prices, "high": prices + 3,
                         "low": prices - 2, "close": prices + 1, "volume": prices / 10})


def archive_payload(frame, mixed_units=False):
    rows = []
    for index, row in frame.iterrows():
        divisor = 1000 if mixed_units and index % 2 else 1_000_000
        stamp = row.timestamp.value // divisor
        rows.append([stamp, row.open, row.high, row.low, row.close, row.volume,
                     stamp + INTERVALS["15m"].value // divisor - 1, 1000, 10, 3, 300, 0])
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-15m.csv", "\n".join(",".join(map(str, row)) for row in rows))
    return output.getvalue()


def install_archive(monkeypatch, frame=None):
    payload = archive_payload(base_candles() if frame is None else frame, mixed_units=True)
    url = data._archive_url("BTC", pd.Timestamp("2025-01-01"), "monthly", "15m")
    calls = []

    def open_url(request, timeout=0):
        calls.append(request.full_url)
        if "data-api.binance.vision" in request.full_url:
            return io.BytesIO(b"[]")
        if "api.kucoin.com" in request.full_url:
            return io.BytesIO(b'{"code":"200000","data":[]}')
        if request.full_url == url:
            return io.BytesIO(payload)
        if request.full_url == url + ".CHECKSUM":
            return io.BytesIO(f"{hashlib.sha256(payload).hexdigest()}  archive.zip\n".encode())
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(data, "urlopen", open_url)
    return calls


def save_frames(directory, frame=None, coin="BTC"):
    frame = base_candles() if frame is None else frame
    for interval in TIMEFRAMES:
        path = directory / interval / f"{coin}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        candles = aggregate_candles(frame, interval, as_of="2025-01-04")
        candles.to_csv(path, index=False)
        sidecar = {"coin": coin, "source": "Binance spot", "symbol": f"{coin}USDT",
                   "quote_currency": "USDT", "interval": interval,
                   "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "rows": len(candles), "status": "downloaded",
                   "first_date": candles.timestamp.min().isoformat(), "last_date": candles.timestamp.max().isoformat()}
        path.with_suffix(".metadata.json").write_text(json.dumps(sidecar), encoding="utf-8")


@pytest.mark.parametrize("interval,expected_count", [("15m", 192), ("1h", 48), ("4h", 12), ("1d", 2)])
def test_aggregation_uses_complete_utc_ohlcv_and_sum_volume(interval, expected_count):
    raw = base_candles()
    result = aggregate_candles(raw, interval, as_of="2025-01-03")
    slots = int(INTERVALS[interval] / INTERVALS["15m"])
    assert len(result) == expected_count
    assert result.iloc[0].open == raw.iloc[0].open
    assert result.iloc[0].high == raw.iloc[:slots].high.max()
    assert result.iloc[0].low == raw.iloc[:slots].low.min()
    assert result.iloc[0].close == raw.iloc[slots - 1].close
    assert result.iloc[0].volume == pytest.approx(raw.iloc[:slots].volume.sum())
    assert result.volume.sum() == pytest.approx(raw.volume.sum())
    assert (result.timestamp == result.timestamp.dt.floor(INTERVALS[interval])).all()


@pytest.mark.parametrize("interval", ["1h", "4h", "1d"])
def test_one_missing_15m_candle_removes_its_entire_higher_bucket(interval):
    raw = base_candles().drop(index=2)
    result = aggregate_candles(raw, interval, as_of="2025-01-03")
    assert pd.Timestamp("2025-01-01", tz="UTC") not in set(result.timestamp)
    assert len(result) == int(pd.Timedelta(days=2) / INTERVALS[interval]) - 1
    assert result.attrs["quality"]["incomplete_buckets_dropped"] == 1


def test_verified_no_trade_coverage_keeps_real_sparse_ohlcv_without_filling():
    raw = base_candles(32).drop(index=[2, 15])
    result = aggregate_candles(raw, "4h", as_of="2025-01-02",
                               no_trade_ranges=[("2025-01-01", "2025-01-01T04:00Z")])
    assert len(result) == 2
    assert result.iloc[0][["open", "high", "low", "close"]].tolist() == [100, 117, 98, 115]
    assert result.iloc[0].volume == pytest.approx(150.3)
    assert result.attrs["quality"]["sparse_buckets_included"] == 1
    assert len(raw) == 30  # The omitted source rows stay absent.


def test_partial_response_coverage_does_not_authorize_sparse_bucket():
    raw = base_candles(32).drop(index=2)
    result = aggregate_candles(raw, "4h", as_of="2025-01-02",
                               no_trade_ranges=[("2025-01-01T00:15Z", "2025-01-01T04:00Z")])
    assert result.timestamp.tolist() == [pd.Timestamp("2025-01-01T04:00Z")]


def test_adjacent_verified_pages_cover_sparse_bucket_but_failed_page_does_not():
    raw = base_candles(32).drop(index=2)
    result = aggregate_candles(raw, "4h", as_of="2025-01-02", no_trade_ranges=[
        ("2025-01-01T02:00Z", "2025-01-01T04:00Z"),
        ("2025-01-01", "2025-01-01T02:00Z")])
    assert len(result) == 2
    failed_page = aggregate_candles(raw, "4h", as_of="2025-01-02", no_trade_ranges=[
        ("2025-01-01", "2025-01-01T02:00Z")])
    assert len(failed_page) == 1


def test_unfinished_candles_are_dropped_at_each_timeframe():
    raw = base_candles()
    cutoff = pd.Timestamp("2025-01-02T05:10:00Z")
    expected = {"15m": 116, "1h": 29, "4h": 7, "1d": 1}
    for interval in TIMEFRAMES:
        result = aggregate_candles(raw, interval, as_of=cutoff)
        assert len(result) == expected[interval]
        assert (result.timestamp + INTERVALS[interval] <= cutoff).all()


def test_duplicate_or_off_grid_base_timestamps_are_rejected():
    raw = base_candles(8)
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_candles(pd.concat([raw, raw.iloc[:1]]), "1h", as_of="2025-01-02")
    raw.loc[0, "timestamp"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="interval grid"):
        aggregate_candles(raw, "1h", as_of="2025-01-02")


def test_download_fetches_only_native_15m_and_preserves_old_daily_history(tmp_path, monkeypatch):
    legacy = tmp_path / "BTC.csv"
    legacy.write_text("legacy daily history", encoding="utf-8")
    calls = install_archive(monkeypatch)
    report = download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)["BTC"]
    assert report["status"] == "downloaded"
    assert report["rows"] == 192
    archive_calls = [url for url in calls if "data.binance.vision" in url]
    assert len(archive_calls) == 2 and all("/15m/" in url for url in archive_calls)
    assert legacy.read_text(encoding="utf-8") == "legacy daily history"
    loaded = load_multiframe_data(tmp_path, ["BTC"], as_of="2025-01-03")["BTC"]
    assert [len(loaded[interval]) for interval in TIMEFRAMES] == [192, 48, 12, 2]
    source_hash = loaded["15m"].attrs["data_sha256"]
    for interval in TIMEFRAMES[1:]:
        provenance = loaded[interval].attrs["provenance"]
        assert provenance["aggregation"]["source_data_sha256"] == source_hash
        assert "volume=sum" in provenance["aggregation"]["formula"]
        assert provenance["aggregation"]["source_archives"]
    assert loaded["15m"].timestamp.iloc[1] == pd.Timestamp("2025-01-01T00:15:00Z")
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: pytest.fail("cached archive made network request"))
    assert download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)["BTC"]["rows"] == 192


def test_download_reports_missing_slots_without_partial_aggregate(tmp_path, monkeypatch):
    install_archive(monkeypatch, base_candles().drop(index=5))
    report = download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)["BTC"]
    assert report["status"] == "partial"
    assert report["timeframes"]["15m"]["unavailable_count"] == 1
    assert report["timeframes"]["1d"]["rows"] == 1
    assert report["timeframes"]["1d"]["unavailable_count"] == 1


def test_load_as_of_keeps_only_fully_closed_candles_at_each_resolution(tmp_path):
    save_frames(tmp_path)
    frames = load_multiframe_data(tmp_path, ["BTC"], as_of="2025-01-02T05:10:00Z")["BTC"]
    assert [len(frames[interval]) for interval in TIMEFRAMES] == [116, 29, 7, 1]
    assert frames["15m"].attrs["quality"]["incomplete_dropped"] == 76


@pytest.mark.parametrize("patch", [
    {"coin": "ETH"}, {"source": "Other exchange"}, {"source": ""},
    {"quote_currency": "EUR"}, {"quote": "BTC"}, {"interval": "1d"},
    {"symbol": "ETHUSDT"}, {"data_sha256": "0" * 64},
    {"timestamp_semantics": "UTC candle close"},
    {"aggregation": {"source_interval": "15m", "source_data_sha256": "0" * 64}},
])
def test_load_rejects_metadata_contradictions(tmp_path, patch):
    save_frames(tmp_path)
    sidecar = tmp_path / "4h" / "BTC.metadata.json"
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar.write_text(json.dumps({**metadata, **patch}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_multiframe_data(tmp_path, ["BTC"], as_of="2025-01-03")


def test_load_requires_all_four_intervals_and_explicit_provenance(tmp_path):
    save_frames(tmp_path)
    (tmp_path / "15m" / "BTC.metadata.json").unlink()
    with pytest.raises(ValueError, match="provenance sidecar"):
        load_multiframe_data(tmp_path, ["BTC"], as_of="2025-01-03")
    (tmp_path / "15m" / "BTC.csv").unlink()
    with pytest.raises(ValueError, match="missing 15m history"):
        load_multiframe_data(tmp_path, ["BTC"], as_of="2025-01-03")


def test_large_missing_intraday_range_has_count_and_bounded_sample():
    raw = base_candles(1)
    quality = data._interval_quality(raw, "15m", pd.Timestamp("2020-01-01", tz="UTC"),
                                     pd.Timestamp("2026-01-01", tz="UTC"))
    expected = int((pd.Timestamp("2026-01-01") - pd.Timestamp("2020-01-01")) / INTERVALS["15m"]) - 1
    assert quality["missing_count"] == expected
    assert len(quality["missing_dates"]) == 100
    assert quality["missing_dates_truncated"] is True


@pytest.mark.parametrize("error", [URLError(ssl.SSLError("temporary TLS failure")), TimeoutError("timeout"),
                                   HTTPError("https://test.invalid/archive", 503, "unavailable", {}, None)])
def test_transient_requests_retry_then_succeed(monkeypatch, error):
    attempts = []

    def open_url(*args, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise error
        return io.BytesIO(b"ok")

    monkeypatch.setattr(data, "urlopen", open_url)
    monkeypatch.setattr(data.time, "sleep", lambda seconds: None)
    assert data._request("https://test.invalid/archive") == b"ok"
    assert len(attempts) == 3


def test_missing_archive_does_not_retry_or_request_every_old_day(tmp_path, monkeypatch):
    requests = []

    def missing(request, timeout=0):
        requests.append(request.full_url)
        if "data-api.binance.vision" in request.full_url:
            return io.BytesIO(b"[]")
        if "api.kucoin.com" in request.full_url:
            return io.BytesIO(b'{"code":"200000","data":[]}')
        raise HTTPError(request.full_url, 404, "missing", {}, None)

    monkeypatch.setattr(data, "urlopen", missing)
    report = download_multiframe_history(["BTC"], "2020-01-01", "2020-02-01", tmp_path)["BTC"]
    assert report["status"] == "failed"
    archive_requests = [url for url in requests if "data.binance.vision" in url]
    assert len(archive_requests) == 1 and "/monthly/" in archive_requests[0]
    assert report["timeframes"]["15m"]["unavailable_count"] == 31 * 96
    assert len(report["timeframes"]["15m"]["unavailable_dates"]) == 100
