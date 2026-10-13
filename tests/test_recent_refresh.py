"""Public recent data append must preserve continuity and bucket volumes."""

import hashlib
import json
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from crypto_forecast import data
from crypto_forecast.timeframes import INTERVALS, TIMEFRAMES
from test_multiframe_data import base_candles, save_frames


def rest_payload(frame):
    return json.dumps([[int(row.timestamp.timestamp()*1000), row.open, row.high,
                        row.low, row.close, row.volume, 0, 0, 0, 0, 0, 0]
                       for row in frame.itertuples()]).encode()


def test_recent_append_rebuilds_complete_buckets_and_provenance(tmp_path, monkeypatch):
    base = base_candles(192).drop(index=[2, 3])
    save_frames(tmp_path, base)
    tail = base_candles(16, "2025-01-03")
    calls = []
    def request(url):
        calls.append(url)
        query = parse_qs(urlparse(url).query)
        assert query["startTime"] == [str(int(tail.timestamp.iloc[0].timestamp()*1000))]
        assert query["interval"] == ["15m"]
        return rest_payload(tail)
    monkeypatch.setattr(data, "_request", request)
    result = data.refresh_multiframe_recent(tmp_path, ["BTC"], as_of="2025-01-03T04:00:00Z")
    assert result["BTC"]["added_rows"] == 16
    frames = data.load_multiframe_data(tmp_path, ["BTC"], as_of="2025-01-03T04:00:00Z")["BTC"]
    assert frames["4h"].volume.iloc[-1] == pytest.approx(tail.volume.sum())
    for interval in TIMEFRAMES:
        metadata = frames[interval].attrs["provenance"]
        assert metadata["recent_rest"][0]["sha256"] == hashlib.sha256(rest_payload(tail)).hexdigest()
        assert metadata["unavailable_count"] == (2 if interval == "15m" else 1)
        assert metadata["status"] == "partial"
        if interval != "15m":
            assert metadata["aggregation"]["source_data_sha256"] == frames["15m"].attrs["data_sha256"]
    # Nothing closed since the preceding refresh; no further requests or writes.
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(data, "_request", lambda _: pytest.fail("current history requested network"))
    assert data.refresh_multiframe_recent(tmp_path, ["BTC"], as_of="2025-01-03T04:10:00Z")["BTC"]["status"] == "current"
    assert all(path.read_bytes() == payload for path, payload in before.items())


@pytest.mark.parametrize("bad", ["schema", "first_gap", "internal_gap", "empty"])
def test_invalid_recent_data_never_replaces_existing_files(tmp_path, monkeypatch, bad):
    save_frames(tmp_path)
    tail = base_candles(16, "2025-01-03")
    if bad == "first_gap":
        tail = tail.iloc[1:]
    elif bad == "internal_gap":
        tail = tail.drop(index=3)
    payload = b'[[1,2]]' if bad == "schema" else b'[]' if bad == "empty" else rest_payload(tail)
    monkeypatch.setattr(data, "_request", lambda _: payload)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        data.refresh_multiframe_recent(tmp_path, ["BTC"], as_of="2025-01-03T04:00:00Z")
    assert all(path.read_bytes() == content for path, content in before.items())
