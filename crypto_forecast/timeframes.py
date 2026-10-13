"""Canonical, UTC-aligned candle intervals used together by the model."""

import pandas as pd


TIMEFRAMES = ("15m", "1h", "4h", "1d")
INTERVALS = {
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


def interval_delta(interval: str) -> pd.Timedelta:
    try:
        return INTERVALS[interval]
    except KeyError as error:
        raise ValueError(f"Unsupported candle interval {interval!r}; choose {TIMEFRAMES}") from error
