"""Exchange selection exercises real downloads with only HTTP bytes replaced."""

import hashlib
import io
import json
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from crypto_forecast import data
from test_multiframe_data import archive_payload, base_candles


def install_markets(monkeypatch, binance, kucoin, *, broken_kucoin=False):
    requests = []

    def open_url(request, timeout=0):
        url = request.full_url
        requests.append(url)
        query = parse_qs(urlparse(url).query)
        if "data-api.binance.vision" in url:
            frame = binance.iloc[:1] if query.get("limit") == ["1"] else binance
            rows = [[int(row.timestamp.timestamp() * 1000), row.open, row.high,
                     row.low, row.close, row.volume, 0, 0, 0, 0, 0, 0]
                    for row in frame.itertuples()]
            return io.BytesIO(json.dumps(rows).encode())
        if "api.kucoin.com" in url:
            left = pd.Timestamp(int(query["startAt"][0]), unit="s", tz="UTC")
            right = pd.Timestamp(int(query["endAt"][0]), unit="s", tz="UTC")
            assert right - left < pd.Timedelta(days=1500 if query["type"] == ["1day"] else 1500 / 96)
            frame = kucoin
            if query["type"] == ["1day"]:
                frame = frame.assign(timestamp=frame.timestamp.dt.floor("D")).drop_duplicates("timestamp")
            frame = frame.loc[(frame.timestamp >= left) & (frame.timestamp <= right)]
            rows = [[str(int(row.timestamp.timestamp())), str(row.open), str(row.close),
                     str(row.high), str(row.low), str(row.volume), "1234"]
                    for row in reversed(list(frame.itertuples()))]
            if broken_kucoin and query["type"] == ["15min"] and right - left > pd.Timedelta(days=1):
                return io.BytesIO(b'{"code":"400100","msg":"invalid symbol"}')
            return io.BytesIO(json.dumps({"code": "200000", "data": rows}).encode())
        if "data.binance.vision" in url:
            payload = archive_payload(binance)
            if url.endswith(".CHECKSUM"):
                return io.BytesIO(f"{hashlib.sha256(payload).hexdigest()}  candles.zip".encode())
            if "/monthly/" in url:
                return io.BytesIO(payload)
        raise HTTPError(url, 404, "missing", {}, None)

    monkeypatch.setattr(data, "urlopen", open_url)
    return requests


def test_hype_uses_older_kucoin_history_and_derives_all_timeframes(tmp_path, monkeypatch):
    install_markets(monkeypatch, base_candles(96, "2025-01-01"),
                    base_candles(228, "2024-12-07T15:00"))
    report = data.download_multiframe_history(["HYPE"], "2024-12-07", "2024-12-10", tmp_path)["HYPE"]
    assert report["source"] == "KuCoin spot"
    assert report["symbol"] == "HYPE-USDT"
    assert report["exchange_comparison"]["KuCoin spot"]["first_date"] == "2024-12-07T15:00:00+00:00"
    assert report["exchange_comparison"]["Binance spot"]["first_date"] == "2025-01-01T00:00:00+00:00"
    frames = data.load_multiframe_data(tmp_path, ["HYPE"], as_of="2024-12-10")["HYPE"]
    assert [len(frames[key]) for key in data.TIMEFRAMES] == [228, 57, 14, 2]
    assert frames["15m"].iloc[0].close == 101
    assert frames["1d"].attrs["provenance"]["source"] == "KuCoin spot"


def test_older_exchange_is_chosen_even_when_both_cover_requested_range(tmp_path, monkeypatch):
    binance = base_candles(288, "2025-01-01")
    kucoin = base_candles(384, "2024-12-31")
    install_markets(monkeypatch, binance, kucoin)
    report = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)["BTC"]
    assert report["source"] == "KuCoin spot"
    frame = pd.read_csv(tmp_path / "15m/BTC.csv")
    assert len(frame) == 192
    assert frame.iloc[0].open == 196  # KuCoin prices, clipped to the requested start.


def test_binance_wins_equal_start_dates(tmp_path, monkeypatch):
    install_markets(monkeypatch, base_candles(), base_candles())
    report = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)["BTC"]
    assert report["source"] == "Binance spot"
    assert report["status"] == "downloaded"


def test_kucoin_no_tick_slots_do_not_discard_first_observed_four_hour_candle(tmp_path, monkeypatch):
    install_markets(monkeypatch, base_candles(96, "2025-01-03"),
                    base_candles(192).drop(index=[2, 15]))
    report = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)["BTC"]
    frames = data.load_multiframe_data(tmp_path, ["BTC"], as_of="2025-01-03")["BTC"]
    assert report["source"] == "KuCoin spot"
    assert len(frames["15m"]) == 190
    assert frames["4h"].timestamp.iloc[0] == pd.Timestamp("2025-01-01T00:00Z")
    assert frames["4h"].iloc[0][["open", "high", "low", "close"]].tolist() == [100, 117, 98, 115]
    assert frames["4h"].iloc[0].volume == pytest.approx(150.3)
    assert len(frames["1d"]) == 2
    assert report["timeframes"]["4h"]["aggregation"]["no_trade_policy"]


def test_kucoin_coverage_requires_successful_matching_bounded_manifests():
    manifest = {"source": "KuCoin spot", "interval": "15m", "sha256": "a" * 64, "rows": 16,
                "url": "https://api.kucoin.com/api/v1/market/candles?symbol=BTC-USDT&type=15min&startAt=1735689600&endAt=1735703999"}
    metadata = {"source": "KuCoin spot", "symbol": "BTC-USDT", "historical_rest": [manifest]}
    assert data._kucoin_no_trade_ranges(metadata) == [(pd.Timestamp("2025-01-01T00:00Z"), pd.Timestamp("2025-01-01T04:00Z"))]
    for changed in [{"source": "Binance spot"}, {"symbol": "LTC-USDT"}, {"historical_rest": []},
                    {"historical_rest": [{**manifest, "sha256": ""}]},
                    {"historical_rest": [{**manifest, "rows": 0}]}]:
        assert data._kucoin_no_trade_ranges({**metadata, **changed}) == []


def test_incremental_kucoin_does_not_redownload_verified_no_tick_slots(tmp_path, monkeypatch):
    install_markets(monkeypatch, base_candles(96, "2025-01-03"), base_candles().drop(index=[2, 15]))
    data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: pytest.fail("verified no-tick slots requested again"))
    report = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path / "stage",
                                             existing_data_dir=tmp_path)["BTC"]
    assert report["reused_rows"] == 190
    assert report["fetched_rows"] == 0
    assert report["timeframes"]["4h"]["rows"] == 12


def test_incomplete_binance_falls_back_to_better_kucoin_coverage(tmp_path, monkeypatch):
    install_markets(monkeypatch, base_candles().drop(index=5), base_candles())
    report = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)["BTC"]
    assert report["source"] == "KuCoin spot"
    assert report["status"] == "downloaded"
    assert report["timeframes"]["1d"]["rows"] == 2


def test_failed_preferred_download_falls_back_without_mixing_sources(tmp_path, monkeypatch):
    install_markets(monkeypatch, base_candles(192, "2025-01-02"),
                    base_candles(288, "2025-01-01"), broken_kucoin=True)
    report = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-04", tmp_path)["BTC"]
    assert report["source"] == "Binance spot"
    assert report["rows"] == 192
    assert report["exchange_comparison"]["KuCoin spot"]["download_errors"]


def test_kucoin_paginates_and_reuses_verified_cache(tmp_path, monkeypatch):
    requests = install_markets(monkeypatch, base_candles(96, "2025-02-01"),
                               base_candles(1632, "2025-01-01"))
    report = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-18", tmp_path)["BTC"]
    assert report["rows"] == 1632
    assert report["timeframes"]["1d"]["rows"] == 17
    assert len(report["timeframes"]["15m"]["historical_rest"]) == 2
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: pytest.fail("cache made network request"))
    again = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-18", tmp_path)["BTC"]
    assert again["rows"] == 1632
    assert requests


def test_kucoin_recent_refresh_retains_source_and_rebuilds_hashes(tmp_path, monkeypatch):
    install_markets(monkeypatch, base_candles(96, "2025-02-01"), base_candles(288))
    data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)
    report = data.refresh_multiframe_recent(tmp_path, ["BTC"], as_of="2025-01-04")["BTC"]
    assert report["added_rows"] == 96
    loaded = data.load_multiframe_data(tmp_path, ["BTC"], as_of="2025-01-04")["BTC"]
    assert len(loaded["1d"]) == 3
    assert loaded["15m"].attrs["provenance"]["source"] == "KuCoin spot"


def test_legacy_daily_download_also_selects_older_exchange(tmp_path, monkeypatch):
    install_markets(monkeypatch, base_candles(96, "2025-02-01"), base_candles())
    report = data.download_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)["BTC"]
    assert report["source"] == "KuCoin spot"
    assert len(data.load_data(tmp_path, ["BTC"], as_of="2025-01-03")["BTC"]) == 2


def test_kucoin_provenance_rejects_other_assets(tmp_path, monkeypatch):
    install_markets(monkeypatch, base_candles(96, "2025-02-01"), base_candles())
    data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)
    for interval in data.TIMEFRAMES:
        path = tmp_path / interval / "BTC.metadata.json"
        metadata = json.loads(path.read_text())
        metadata["symbol"] = "ETH-USDT"
        path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="symbol"):
        data.load_multiframe_data(tmp_path, ["BTC"], as_of="2025-01-03")


@pytest.mark.parametrize("bad", ["duplicate", "schema"])
def test_bad_kucoin_refresh_preserves_every_existing_candle_file(tmp_path, monkeypatch, bad):
    install_markets(monkeypatch, base_candles(96, "2025-02-01"), base_candles(288))
    data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)
    tail = base_candles(96, "2025-01-03")
    if bad == "gap":
        tail = tail.drop(index=3)
    elif bad == "trailing_gap":
        tail = tail.iloc[:-1]
    elif bad == "duplicate":
        tail = pd.concat([tail, tail.iloc[:1]])
    payload = {"code": "200000", "data": [[str(int(row.timestamp.timestamp())), row.open, row.close,
                                             row.high, row.low, row.volume, "1234"] for row in tail.itertuples()]}
    if bad == "schema":
        payload["data"] = [["1", "2"]]
    monkeypatch.setattr(data, "_request", lambda _: json.dumps(payload).encode())
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        data.refresh_multiframe_recent(tmp_path, ["BTC"], as_of="2025-01-04")
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize("omitted", [[0, 3, 95], list(range(96))])
def test_kucoin_refresh_accepts_checked_no_tick_slots_without_filling(tmp_path, monkeypatch, omitted):
    install_markets(monkeypatch, base_candles(96, "2025-02-01"), base_candles(288).drop(index=[5]))
    data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)
    tail = base_candles(96, "2025-01-03").drop(index=omitted)
    payload = {"code": "200000", "data": [[str(int(row.timestamp.timestamp())), row.open, row.close,
                                             row.high, row.low, row.volume, "0"] for row in tail.itertuples()]}
    monkeypatch.setattr(data, "_request", lambda _: json.dumps(payload).encode())
    result = data.refresh_multiframe_recent(tmp_path, ["BTC"], as_of="2025-01-04")["BTC"]
    assert result["added_rows"] == 96 - len(omitted)
    frames = data.load_multiframe_data(tmp_path, ["BTC"], as_of="2025-01-04")["BTC"]
    assert frames["4h"].timestamp.iloc[0] == pd.Timestamp("2025-01-01T00:00Z")
    assert len(frames["15m"]) == 287 - len(omitted)
    assert len(frames["1d"]) == (2 if len(omitted) == 96 else 3)


def test_kucoin_no_tick_source_intervals_remain_absent_but_derived_candles_survive(tmp_path, monkeypatch):
    install_markets(monkeypatch, base_candles(96, "2025-02-01"), base_candles().drop(index=5))
    report = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)["BTC"]
    assert report["source"] == "KuCoin spot"
    assert report["timeframes"]["15m"]["unavailable_count"] == 1
    assert report["timeframes"]["1d"]["rows"] == 2
    assert report["status"] == "partial"


def test_failed_comparison_still_uses_valid_binance_archives(tmp_path, monkeypatch):
    install_markets(monkeypatch, base_candles(), base_candles(0))
    original = data.urlopen
    def unavailable(request, timeout=0):
        if "api.kucoin.com" in request.full_url:
            raise HTTPError(request.full_url, 503, "temporarily unavailable", {}, None)
        return original(request, timeout)
    monkeypatch.setattr(data, "urlopen", unavailable)
    monkeypatch.setattr(data.time, "sleep", lambda seconds: None)
    report = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)["BTC"]
    assert report["source"] == "Binance spot"
    assert report["status"] == "downloaded"
    assert report["exchange_comparison"]["KuCoin spot"]["status"] == "error"


def test_first_kucoin_daily_candle_without_intraday_data_does_not_hide_later_history(tmp_path, monkeypatch):
    install_markets(monkeypatch, base_candles(96, "2025-02-01"), base_candles(288))
    original = data.urlopen
    def missing_first_day(request, timeout=0):
        query = parse_qs(urlparse(request.full_url).query)
        if query.get("type") == ["15min"] and query.get("startAt") == ["1735689600"]:
            return io.BytesIO(b'{"code":"200000","data":[]}')
        return original(request, timeout)
    monkeypatch.setattr(data, "urlopen", missing_first_day)
    comparison = data._spot_first_dates("BTC", data._utc("2025-03-01"), tmp_path)
    assert comparison["KuCoin spot"]["first_date"] == "2025-01-02T00:00:00+00:00"


def test_sparse_older_history_does_not_beat_a_complete_post_listing_alternative(tmp_path, monkeypatch):
    install_markets(monkeypatch, base_candles(192, "2025-01-02"), base_candles(1))
    report = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-04", tmp_path)["BTC"]
    assert report["source"] == "Binance spot"
    assert report["timeframes"]["1d"]["rows"] == 2


def test_unusable_source_switch_preserves_existing_complete_history(tmp_path, monkeypatch):
    install_markets(monkeypatch, base_candles(), base_candles(0))
    data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)
    before = {path: path.read_bytes() for interval in data.TIMEFRAMES
              for path in (tmp_path / interval).iterdir()}
    install_markets(monkeypatch, base_candles().drop(index=[5, 101]),
                    base_candles(1, "2024-12-31"))
    report = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path, refresh=True)["BTC"]
    assert report["status"] == "failed"
    assert all(path.read_bytes() == payload for path, payload in before.items())
    frames = data.load_multiframe_data(tmp_path, ["BTC"], as_of="2025-01-03")["BTC"]
    assert frames["15m"].attrs["provenance"]["source"] == "Binance spot"
