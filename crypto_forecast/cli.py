"""Reproducible research commands; never requires exchange account credentials."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pandas as pd

from .data import aggregate_candles, download_history, download_multiframe_history, refresh_multiframe_recent
from .pipeline import TrainingConfig, predict, train
from .model import MODEL_SPECS
from .timeframes import TIMEFRAMES

DEFAULT_COINS = ["BTC", "ETH", "SOL", "XRP", "ADA"]


def _settings(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")
    unknown = set(config) - set(TrainingConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    config.setdefault("coins", DEFAULT_COINS.copy())
    for name in TrainingConfig.__dataclass_fields__:
        override = getattr(args, name, None)
        if override is not None:
            config[name] = override
    return TrainingConfig(**config)


def _demo(args):
    run_dir = Path(args.run_dir)
    data_dir = run_dir / "synthetic_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    multi = args.input_mode == "multiframe"
    dates = pd.date_range("2022-01-01", periods=400*96 if multi else 850,
                          tz="UTC", freq="15min" if multi else "D")
    volatility = 0.018/np.sqrt(96) if multi else 0.018
    common = rng.normal(0, volatility, len(dates))
    for coin, initial in (("BTC", 35000), ("ETH", 2000)):
        close = initial * np.exp(np.cumsum(common + rng.normal(0.0001/(96 if multi else 1),
                                             volatility*0.4, len(dates))))
        opening = np.concatenate(([initial], close[:-1]))
        frame = pd.DataFrame({"timestamp": dates, "open": opening,
                      "high": np.maximum(opening, close)*1.012,
                      "low": np.minimum(opening, close)*0.988, "close": close,
                      "volume": rng.lognormal(9, 0.4, len(dates))})
        for interval in TIMEFRAMES if multi else ("1d",):
            output = data_dir / interval if multi else data_dir
            output.mkdir(parents=True, exist_ok=True)
            candles = aggregate_candles(frame, interval, coin=coin) if multi else frame
            csv_path = output / f"{coin}.csv"
            candles.to_csv(csv_path, index=False)
            (output / f"{coin}.metadata.json").write_text(json.dumps({
                "source": "synthetic_demo", "coin": coin, "quote_currency": "USDT",
                "interval": interval, "seed": 42,
                "data_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}), encoding="utf-8")
    config = TrainingConfig(coins=["BTC", "ETH"], start="2022-01-01",
                            data_dir=str(data_dir), run_dir=str(run_dir), epochs=args.epochs,
                            hidden_size=16, device="cpu", input_mode=args.input_mode,
                            model_type=args.model_type,
                            kronos_timeframe="4h" if multi else "1d",
                            batch_size=8 if args.model_type in {"kronos-small", "kronos-base"} else 128,
                            learning_rate=1e-5 if args.model_type in {"kronos-small", "kronos-base"} else .001,
                            origin_interval="4h")
    print("SYNTHETIC DEMONSTRATION: verifies the pipeline; provides no market evidence.")
    return train(config, overwrite=args.overwrite, data_kind="synthetic_demo")


def parser():
    root = argparse.ArgumentParser(description="Train and validate deep learning spot crypto price forecasts.")
    commands = root.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download public spot OHLCV for all four candle timeframes.")
    training = commands.add_parser("train", help="Train, freeze, and evaluate on chronological historical splits.")
    for command in (download, training):
        command.add_argument("--config", help="JSON configuration; command-line options override its values.")
        command.add_argument("--coins", nargs="+", help="Selected coin symbols, e.g. BTC ETH SOL.")
        command.add_argument("--start", help="First daily open, inclusive, YYYY-MM-DD.")
        command.add_argument("--end", help="Last UTC boundary, exclusive, YYYY-MM-DD (default: today UTC).")
        command.add_argument("--data-dir")
        command.add_argument("--input-mode", choices=("multiframe", "daily"))
        command.add_argument("--origin-interval", choices=TIMEFRAMES)
    download.add_argument("--refresh", action="store_true", help="Re-download archives instead of using cached copies.")
    download.add_argument("--latest", action="store_true", help="Append current completed 15m candles from public REST data.")
    training.add_argument("--run-dir")
    training.add_argument("--model-type", choices=tuple(MODEL_SPECS), help="Architecture; Kronos-small fine-tunes pretrained weights.")
    training.add_argument("--kronos-timeframe", choices=TIMEFRAMES)
    training.add_argument("--kronos-prediction-length", type=int,
                          help="Future candles per training target (default 24); input plus target minus one must fit 512 tokens.")
    training.add_argument("--kronos-model-path", help="Local pretrained predictor directory or official Hugging Face ID.")
    training.add_argument("--kronos-tokenizer-path", help="Local pretrained tokenizer directory or official Hugging Face ID.")
    training.add_argument("--kronos-training-stage", choices=("tokenizer", "predictor", "sequential"),
                          help="Train a reusable tokenizer, train a predictor with a saved tokenizer, or run both stages (default sequential).")
    training.add_argument("--kronos-tokenizer-run-dir", help="Run folder containing the saved tokenizer to freeze during predictor training.")
    training.add_argument("--kronos-tokenizer-batch-size", type=int,
                          help="Tokenizer batch size; defaults to --batch-size when omitted.")
    training.add_argument("--kronos-sample-count", type=int, help="Forecast paths to average (default 1).")
    training.add_argument("--kronos-eval-stride", type=int, help="Evaluate every N eligible origins per coin (default 6).")
    training.add_argument("--kronos-train-stride", type=int, help="Train on every N eligible origins per coin (default 6).")
    training.add_argument("--kronos-preparation-workers", type=int,
                          help="CPU preparation workers: 0 automatic, 1 serial, or 2-4 processes (default 0).")
    training.add_argument("--kronos-finetune-tokenizer", action=argparse.BooleanOptionalAction, default=None,
                          help="Fine-tune tokenizer before predictor (default enabled); --no-kronos-finetune-tokenizer skips it.")
    training.add_argument("--kronos-tokenizer-epochs", type=int, help="Maximum tokenizer epochs (default 5).")
    training.add_argument("--kronos-tokenizer-patience", type=int, help="Tokenizer early-stop patience (default 2).")
    training.add_argument("--kronos-tokenizer-learning-rate", type=float, help="Tokenizer peak learning rate (default 0.0002).")
    training.add_argument("--lookback", type=int)
    training.add_argument("--hidden-size", type=int)
    training.add_argument("--epochs", type=int)
    training.add_argument("--patience", type=int)
    training.add_argument("--batch-size", type=int)
    training.add_argument("--learning-rate", type=float)
    from .learning_rate import POLICIES
    for prefix in ("", "kronos-tokenizer-"):
        training.add_argument(f"--{prefix}lr-scheduler", choices=POLICIES)
        training.add_argument(f"--{prefix}lr-min-ratio", type=float, help="Minimum rate as a fraction of the peak (default 0.1).")
        training.add_argument(f"--{prefix}lr-warmup-fraction", type=float, help="Fraction of planned updates used for warmup.")
        training.add_argument(f"--{prefix}lr-factor", type=float, help="Step/plateau reduction multiplier (default 0.5).")
        training.add_argument(f"--{prefix}lr-step-epochs", type=int)
        training.add_argument(f"--{prefix}lr-patience", type=int, help="Plateau tolerance in non-improving validation epochs.")
    training.add_argument("--train-fraction", type=float)
    training.add_argument("--validation-fraction", type=float)
    training.add_argument("--seed", type=int)
    training.add_argument("--device", choices=("auto", "cpu", "cuda"))
    training.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing experiment.")
    prediction = commands.add_parser("predict", help="Forecast from the latest complete selected-coin candles.")
    prediction.add_argument("--run-dir", default="runs/multiframe-coins")
    prediction.add_argument("--data-dir", default="data")
    prediction.add_argument("--coins", nargs="+")
    prediction.add_argument("--allow-stale", action="store_true", help="Permit explicitly labeled historical data.")
    prediction.add_argument("--max-age-days", type=float, help="Freshness limit; default 8 hours for multiframe, 2 days for daily.")
    demo = commands.add_parser("demo", help="Run a clearly labeled synthetic end-to-end demonstration.")
    demo.add_argument("--run-dir", default="runs/multiframe-demo")
    demo.add_argument("--input-mode", choices=("multiframe", "daily"), default="multiframe")
    demo.add_argument("--model-type", choices=tuple(MODEL_SPECS), default="gru")
    demo.add_argument("--epochs", type=int, default=3)
    demo.add_argument("--overwrite", action="store_true")
    dashboard = commands.add_parser("dashboard", help="Open a local browser training dashboard with live progress.")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument("--workspace", default=".", help="Project folder containing data and runs.")
    setup = commands.add_parser("kronos-setup", help="Download an official Kronos predictor and base tokenizer for local use.")
    setup.add_argument("--model", choices=("kronos-small", "kronos-base"), default="kronos-small")
    setup.add_argument("--model-dir", help="Local destination (default: models/selected-model).")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "kronos-setup":
            from .kronos_setup import download_pretrained
            download_pretrained(args.model_dir, model=args.model)
        elif args.command == "dashboard":
            from .dashboard import serve_dashboard
            serve_dashboard(args.workspace, args.port)
        elif args.command == "demo":
            _demo(args)
        elif args.command == "predict":
            forecasts = predict(args.run_dir, args.data_dir, args.coins,
                                allow_stale=args.allow_stale, max_age_days=args.max_age_days)
            for row in forecasts:
                stale = " [STALE HISTORICAL INPUT]" if row["is_stale"] else ""
                horizon = (f"{row['horizon_candles']} {row['timeframe']} candles" if "horizon_candles" in row
                           else f"{row['horizon_days']}d")
                print(f"{row['coin']:5} {horizon}: {row['forecast_price_usdt']:.6g} USDT "
                      f"({row['change_pct']:+.2f}%), as of {row['as_of']}{stale}")
        else:
            config = _settings(args)
            config.validate()
            if args.command == "download":
                end = config.end or datetime.now(timezone.utc).date().isoformat()
                downloader = download_multiframe_history if config.input_mode == "multiframe" else download_history
                result = downloader(config.coins, config.start, end, config.data_dir, refresh=args.refresh)
                if config.input_mode == "multiframe" and (args.latest or config.end is None):
                    print("Appending current completed 15m candles and rebuilding the larger timeframes", flush=True)
                    recent = refresh_multiframe_recent(config.data_dir, config.coins)
                    for coin in result:
                        result[coin]["recent_refresh"] = recent[coin]
                output = Path(config.data_dir) / "download_report.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
                for coin, info in result.items():
                    print(f"{coin}: {info.get('status')} ({info.get('rows', info.get('row_count', '?'))} "
                          f"{'15m source' if config.input_mode == 'multiframe' else 'daily'} candles, "
                          f"{info.get('source', 'public spot data')})")
                print(f"Acquisition details: {output.resolve()}")
                if any(info.get("status") == "failed" for info in result.values()):
                    return 1
            else:
                train(config, overwrite=args.overwrite)
        return 0
    except (ValueError, OSError, URLError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
