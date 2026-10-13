"""Data boundary tests: real normalization/caching, fake only network bytes."""

import hashlib
import io
import json
from pathlib import Path
from urllib.error import HTTPError
import zipfile

import pandas as pd
import pytest

from crypto_forecast import data as data_module
from crypto_forecast.data import download_history, load_data


def write_csv(path, timestamps, **overrides):
    rows = {"timestamp": timestamps, "open": [10.0] * len(timestamps),
            "high": [12.0] * len(timestamps), "low": [9.0] * len(timestamps),
            "close": [11.0] * len(timestamps), "volume": [100.0] * len(timestamps)}
    rows.update(overrides)
    pd.DataFrame(rows).to_csv(path, index=False)


def archive_bytes(rows):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-1d.csv", "\n".join(
            ",".join(map(str, [timestamp, 10, 12, 9, 11, 100,
                              timestamp + unit - 1, 1100, 10, 60, 660, 0]))
            for timestamp, unit in rows))
    return output.getvalue()


def fake_archives(monkeypatch, archives, bad_checksum=False):
    def open_url(request, timeout=0):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        if "data-api.binance.vision" in url:
            return io.BytesIO(b"[]")
        if "api.kucoin.com" in url:
            return io.BytesIO(b'{"code":"200000","data":[]}')
        base = url.removesuffix(".CHECKSUM")
        if base not in archives:
            raise HTTPError(url, 404, "Not Found", {}, None)
        payload = archives[base]
        if url.endswith(".CHECKSUM"):
            digest = "0" * 64 if bad_checksum else hashlib.sha256(payload).hexdigest()
            payload = f"{digest}  {base.rsplit('/', 1)[-1]}\n".encode()
        return io.BytesIO(payload)
    monkeypatch.setattr("crypto_forecast.data.urlopen", open_url)


def test_load_sorts_drops_incomplete_and_preserves_gaps(tmp_path):
    write_csv(tmp_path / "BTC.csv", ["2026-01-04", "2026-01-03", "2026-01-01"])
    frames = load_data(tmp_path, ["BTC"], as_of="2026-01-04T12:00:00Z")
    frame = frames["BTC"]
    assert frame.timestamp.tolist() == [pd.Timestamp("2026-01-01", tz="UTC"),
                                        pd.Timestamp("2026-01-03", tz="UTC")]
    assert frame.attrs["quality"]["missing_dates"] == ["2026-01-02"]
    assert frame.attrs["quality"]["incomplete_dropped"] == 1


def test_load_normalizes_mixed_ms_us_epochs(tmp_path):
    write_csv(tmp_path / "BTC.csv", [1735689600000, 1735776000000000])
    frame = load_data(tmp_path, ["BTC"], as_of="2025-01-03")["BTC"]
    assert frame.timestamp.tolist() == [pd.Timestamp("2025-01-01", tz="UTC"),
                                        pd.Timestamp("2025-01-02", tz="UTC")]


@pytest.mark.parametrize("metadata", [
    {"coin": "ETH", "symbol": "ETHUSDT", "source": "Binance spot"},
    {"coin": "BTC", "symbol": "ETHUSDT", "source": "Binance spot"},
    {"symbol": "ETHUSDT"},
    {"source": "Binance spot", "symbol": "BTCIRT"},
    {"quote": "EUR"},
    {"quote_currency": "USDT", "quote": "BTC"},
    {"interval": "1h"},
])
def test_load_rejects_conflicting_source_identity_even_with_correct_hash(tmp_path, metadata):
    path = tmp_path / "BTC.csv"
    write_csv(path, ["2025-01-01"])
    metadata = {**metadata, "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / "BTC.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        load_data(tmp_path, ["BTC"], as_of="2025-01-02")


def test_load_accepts_supplemental_exchange_instrument_and_quote_alias(tmp_path):
    write_csv(tmp_path / "XAUT.csv", ["2025-01-01"])
    metadata = {"coin": "xaut", "symbol": "tXAUT:UST", "source": "Bitfinex spot",
                "quote": "USDT", "interval": "1d"}
    (tmp_path / "XAUT.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    frame = load_data(tmp_path, ["XAUT"], as_of="2025-01-02")["XAUT"]
    assert frame.attrs["coin"] == "XAUT"
    assert frame.attrs["provenance"]["symbol"] == "tXAUT:UST"


@pytest.mark.parametrize("timestamps,overrides", [
    (["2025-01-01", "2025-01-01"], {}),
    (["2025-01-01"], {"close": [float("inf")]}),
    (["2025-01-01"], {"volume": [-1]}),
    (["2025-01-01"], {"close": [15]}),
    (["2025-01-01T12:00:00Z"], {}),
])
def test_load_rejects_invalid_data(tmp_path, timestamps, overrides):
    write_csv(tmp_path / "BTC.csv", timestamps, **overrides)
    with pytest.raises(ValueError):
        load_data(tmp_path, ["BTC"], as_of="2025-02-01")


@pytest.mark.parametrize("coin", ["../BTC", "USDT", "", "BTC/USDT", "CON"])
def test_coin_validation_blocks_unsafe_paths_and_quote_target(tmp_path, coin):
    with pytest.raises(ValueError):
        load_data(tmp_path, [coin])


def test_download_verifies_checksum_filters_end_and_reuses_cache(tmp_path, monkeypatch):
    url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2025-01.zip"
    payload = archive_bytes([(1735689600000, 86400000),
                             (1735776000000000, 86400000000),
                             (1735862400000000, 86400000000)])
    fake_archives(monkeypatch, {url: payload})
    metadata = download_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)
    assert metadata["BTC"]["status"] == "downloaded"
    frame = load_data(tmp_path, ["BTC"], as_of="2025-01-03")["BTC"]
    assert frame.timestamp.dt.strftime("%Y-%m-%d").tolist() == ["2025-01-01", "2025-01-02"]
    saved = json.loads((tmp_path / "BTC.metadata.json").read_text())
    assert saved["archives"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert saved["data_sha256"] == hashlib.sha256((tmp_path / "BTC.csv").read_bytes()).hexdigest()
    monkeypatch.setattr("crypto_forecast.data.urlopen", lambda *args, **kwargs: pytest.fail("cache made a request"))
    assert download_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)["BTC"]["rows"] == 2


def test_recent_missing_month_falls_back_to_daily_and_reports_missing_days(tmp_path, monkeypatch):
    original_utc = data_module._utc
    monkeypatch.setattr(data_module, "_utc", lambda value=None: original_utc("2025-02-02") if value is None else original_utc(value))
    url = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2025-01-01.zip"
    fake_archives(monkeypatch, {url: archive_bytes([(1735689600000, 86400000)])})
    metadata = download_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path)["BTC"]
    assert metadata["status"] == "partial"
    assert metadata["rows"] == 1
    assert any("2025-01-02.zip" in item for item in metadata["missing_archives"])
    assert metadata["unavailable_dates"] == ["2025-01-02"]


def test_old_missing_month_reports_absence_without_daily_request_storm(tmp_path, monkeypatch):
    def missing_month(request, timeout=0):
        url = request.full_url
        if "/daily/" in url:
            pytest.fail("Old pre-listing months must not request every daily archive")
        raise HTTPError(url, 404, "Not Found", {}, None)
    monkeypatch.setattr(data_module, "urlopen", missing_month)
    metadata = download_history(["BTC"], "2020-01-01", "2020-02-01", tmp_path)["BTC"]
    assert metadata["status"] == "failed"
    assert len(metadata["missing_archives"]) == 1
    assert len(metadata["unavailable_dates"]) == 31


def test_missing_official_checksum_is_failure_not_a_listing_gap(tmp_path, monkeypatch):
    payload = archive_bytes([(1735689600000, 86400000)])
    def without_checksum(request, timeout=0):
        url = request.full_url
        if url.endswith(".CHECKSUM"):
            raise HTTPError(url, 404, "Not Found", {}, None)
        return io.BytesIO(payload)
    monkeypatch.setattr(data_module, "urlopen", without_checksum)
    metadata = download_history(["BTC"], "2025-01-01", "2025-01-02", tmp_path)["BTC"]
    assert metadata["status"] == "failed"
    assert "checksum" in " ".join(metadata["errors"]).lower()
    assert not metadata["missing_archives"]


def test_checksum_failure_does_not_replace_existing_history(tmp_path, monkeypatch):
    existing = tmp_path / "BTC.csv"
    write_csv(existing, ["2024-01-01"])
    original = existing.read_bytes()
    url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2025-01.zip"
    fake_archives(monkeypatch, {url: archive_bytes([(1735689600000, 86400000)])}, bad_checksum=True)
    metadata = download_history(["BTC"], "2025-01-01", "2025-01-02", tmp_path)["BTC"]
    assert metadata["status"] == "failed"
    assert "checksum" in " ".join(metadata["errors"]).lower()
    assert existing.read_bytes() == original
