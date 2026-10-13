"""Unavailable exchange responses must not prevent later coverage repair."""
import io
import json
from urllib.parse import parse_qs, urlparse
from urllib.error import HTTPError

import pandas as pd

from crypto_forecast import data
from test_exchange_selection import install_markets
from test_multiframe_data import base_candles, save_frames


def test_empty_kucoin_gap_response_is_retried_on_next_download(tmp_path, monkeypatch):
    full = base_candles()
    save_frames(tmp_path, full.drop(index=5))
    for interval in data.TIMEFRAMES:
        path = tmp_path / interval / "BTC.metadata.json"
        metadata = json.loads(path.read_text())
        metadata.update(source="KuCoin spot", symbol="BTC-USDT", requested_start="2025-01-01T00:00:00+00:00",
                        exchange_comparison={"KuCoin spot": {"first_date": "2025-01-01T00:00:00+00:00"}})
        path.write_text(json.dumps(metadata), encoding="utf-8")
    attempts = []
    def open_url(request, timeout=0):
        query = parse_qs(urlparse(request.full_url).query)
        assert query["startAt"] == ["1735694100"]
        attempts.append(1)
        row = full.iloc[5]
        rows = [] if len(attempts) == 1 else [[str(int(row.timestamp.timestamp())), str(row.open),
                str(row.close), str(row.high), str(row.low), str(row.volume), "0"]]
        return io.BytesIO(json.dumps({"code": "200000", "data": rows}).encode())
    monkeypatch.setattr(data, "urlopen", open_url)
    first = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path,
                                            existing_data_dir=tmp_path)["BTC"]
    assert first["rows"] == 191
    second = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path,
                                             existing_data_dir=tmp_path)["BTC"]
    assert second["rows"] == 192
    assert len(attempts) == 2


def test_fresh_binance_download_does_not_request_known_prelisting_archives(tmp_path, monkeypatch):
    install_markets(monkeypatch, base_candles(), base_candles())
    original_open = data.urlopen
    requests = []
    def open_url(request, timeout=0):
        requests.append(request.full_url)
        if "data.binance.vision/data/spot/" in request.full_url and "2025-01" not in request.full_url:
            raise HTTPError(request.full_url, 404, "before listing", {}, None)
        return original_open(request, timeout=timeout)
    monkeypatch.setattr(data, "urlopen", open_url)
    report = data.download_multiframe_history(["BTC"], "2017-01-01", "2025-01-03", tmp_path)["BTC"]
    archive_urls = [url for url in requests if "data.binance.vision/data/spot/" in url]
    assert archive_urls
    assert all("2025-01" in url for url in archive_urls)
    assert report["requested_start"] == "2017-01-01T00:00:00+00:00"
    assert report["rows"] == 192
