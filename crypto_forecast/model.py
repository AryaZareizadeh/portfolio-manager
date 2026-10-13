"""Small pooled recurrent model and serializable training-only normalizers."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


MODEL_SPECS = {
    "gru": {
        "id": "gru", "name": "Multi-timeframe GRU",
        "description": "Four gated recurrent encoders combine candle price and volume signals.",
        "n_features": 11, "embedding_size": 8, "dropout": 0.15, "encoder_gates": 3,
    },
    "lstm": {
        "id": "lstm", "name": "Multi-timeframe LSTM",
        "description": "Four recurrent encoders use separate memory cells to combine price and volume signals.",
        "n_features": 11, "embedding_size": 8, "dropout": 0.15, "encoder_gates": 4,
    },
    "kronos-small": {
        "id": "kronos-small", "name": "Kronos-small (fine-tune)",
        "description": "Fine-tune the pretrained candle Transformer on a selected candle interval and forecast length.",
        "n_features": 6, "parameters": 24700000, "max_context": 512,
        "backend": "kronos", "pretrained_model": "NeoQuasar/Kronos-small",
    },
    "kronos-base": {
        "id": "kronos-base", "name": "Kronos-base (fine-tune)",
        "description": "Fine-tune the pretrained base candle Transformer on a selected candle interval and forecast length.",
        "n_features": 6, "parameters": 102300000, "max_context": 512,
        "backend": "kronos", "pretrained_model": "NeoQuasar/Kronos-base",
    },
}


@dataclass
class Normalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Normalizer":
        values = np.asarray(values, dtype=np.float64)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Normalizer requires finite training observations.")
        return cls(values.mean(axis=0), np.maximum(values.std(axis=0), 1e-6))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, value: dict) -> "Normalizer":
        return cls(np.asarray(value["mean"]), np.asarray(value["scale"]))

    @classmethod
    def fit_multiframe(cls, samples) -> "Normalizer":
        """Fit each branch independently, without copying the full training set.

        Parallel variance updates aggregate the bar observations in each window.
        Only the supplied training samples contribute to these statistics.
        """
        count, mean, m2 = 0, None, None
        for sample in samples:
            values = np.asarray(sample.x, dtype=np.float64)
            if values.ndim != 3 or not np.isfinite(values).all():
                raise ValueError("Multiframe normalizer requires finite [timeframe, bars, features] inputs.")
            size = values.shape[1]
            batch_mean = values.mean(axis=1)
            batch_m2 = ((values - batch_mean[:, None, :]) ** 2).sum(axis=1)
            if mean is None:
                mean, m2, count = batch_mean, batch_m2, size
            else:
                delta = batch_mean - mean
                total = count + size
                m2 += batch_m2 + delta**2 * count * size / total
                mean += delta * size / total
                count = total
        if not count:
            raise ValueError("Multiframe normalizer requires training samples.")
        return cls(mean[:, None, :], np.maximum(np.sqrt(m2/count), 1e-6)[:, None, :])


class _ForecastRecurrent(nn.Module):
    def __init__(self, n_features: int, n_coins: int, hidden_size: int, model_type: str):
        super().__init__()
        encoder = nn.GRU if model_type == "gru" else nn.LSTM
        self.encoder = encoder(n_features, hidden_size, batch_first=True)
        self.coin_embedding = nn.Embedding(n_coins, 8)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 8, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x: torch.Tensor, coins: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(x)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return self.head(torch.cat((hidden[-1], self.coin_embedding(coins)), dim=1))


class ForecastGRU(_ForecastRecurrent):
    def __init__(self, n_features: int, n_coins: int, hidden_size: int = 48):
        super().__init__(n_features, n_coins, hidden_size, "gru")


class ForecastLSTM(_ForecastRecurrent):
    def __init__(self, n_features: int, n_coins: int, hidden_size: int = 48):
        super().__init__(n_features, n_coins, hidden_size, "lstm")


class _MultiTimeframeRecurrent(nn.Module):
    """Encode four candle resolutions and fuse them into one joint forecast."""

    def __init__(self, n_features: int, n_coins: int, hidden_size: int, n_timeframes: int,
                 model_type: str):
        super().__init__()
        encoder = nn.GRU if model_type == "gru" else nn.LSTM
        self.encoders = nn.ModuleList([
            encoder(n_features, hidden_size, batch_first=True) for _ in range(n_timeframes)
        ])
        self.coin_embedding = nn.Embedding(n_coins, 8)
        self.head = nn.Sequential(
            nn.Linear(hidden_size*n_timeframes+8, hidden_size*2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size*2, 3),
        )

    def forward(self, x: torch.Tensor, coins: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != len(self.encoders):
            raise ValueError("Expected [batch, four timeframes, bars, features].")
        hidden = []
        for index, encoder in enumerate(self.encoders):
            state = encoder(x[:, index])[1]
            if isinstance(state, tuple):
                state = state[0]
            hidden.append(state[-1])
        return self.head(torch.cat((*hidden, self.coin_embedding(coins)), dim=1))


class MultiTimeframeGRU(_MultiTimeframeRecurrent):
    def __init__(self, n_features: int, n_coins: int, hidden_size: int = 48, n_timeframes: int = 4):
        super().__init__(n_features, n_coins, hidden_size, n_timeframes, "gru")


class MultiTimeframeLSTM(_MultiTimeframeRecurrent):
    def __init__(self, n_features: int, n_coins: int, hidden_size: int = 48, n_timeframes: int = 4):
        super().__init__(n_features, n_coins, hidden_size, n_timeframes, "lstm")


def build_model(n_features: int, n_coins: int, hidden_size: int = 48, *,
                input_mode: str = "multiframe", model_type: str = "gru") -> nn.Module:
    """Construct the architecture recorded in a run's configuration."""
    classes = {("multiframe", "gru"): MultiTimeframeGRU,
               ("multiframe", "lstm"): MultiTimeframeLSTM,
               ("daily", "gru"): ForecastGRU,
               ("daily", "lstm"): ForecastLSTM}
    try:
        model_class = classes[input_mode, model_type]
    except (KeyError, TypeError):
        raise ValueError("Select input_mode daily or multiframe and model_type gru or lstm.") from None
    return model_class(n_features, n_coins, hidden_size)
