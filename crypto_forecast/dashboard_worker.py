"""One isolated dashboard job; progress survives browser reconnects."""

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback

# Windows spawn imports the launching module as __mp_main__. Preparation
# children only need the CPU helper module, so avoid loading Torch there.
if __name__ != "__mp_main__":
    from .dashboard import publish_download, read_json, utc_now, write_json
    from .data import download_multiframe_history, refresh_multiframe_recent
    from .pipeline import TrainingConfig, train


def run_job(path, commands=None):
    path = Path(path)
    job = read_json(path)
    lock = threading.Lock()

    def save():
        job["updated_at"] = utc_now()
        write_json(path, job)

    def log(message):
        with lock:
            job["logs"].append({"timestamp": utc_now(), "message": str(message)})
            job["logs"] = job["logs"][-250:]
            save()
        print(message, flush=True)

    def event(value):
        with lock:
            job["stage"] = value.get("stage", job["stage"])
            job["progress"].update({k: v for k, v in value.items() if k not in {"kind", "stage"}})
            if value["kind"] in ("epoch", "tokenizer_epoch") and "lr_end" in value:
                # The curve retains the epoch mean; the live value is the last
                # rate actually used, not the mean or the next scheduled rate.
                job["progress"]["learning_rate"] = value["lr_end"]
            if value["kind"] == "epoch":
                job["history"].append({k: value[k] for k in ("epoch", "train_loss", "validation_loss", "learning_rate", "lr_end", "lr_next") if k in value})
            elif value["kind"] == "tokenizer_epoch":
                job.setdefault("tokenizer_history", []).append(
                    {k: value[k] for k in ("epoch", "train_loss", "validation_loss", "learning_rate", "lr_end", "lr_next") if k in value})
            save()

    last_download_save = 0.0

    def download_progress(value):
        nonlocal last_download_save
        with lock:
            coin = value["coin"]
            row = job["progress"]["coin_progress"][coin]
            phase_changed = row.get("phase") != value.get("phase")
            total_changed = "total" in value and row.get("total") != value["total"]
            row.update({key: value[key] for key in ("phase", "completed", "total", "source", "detail") if key in value})
            row["status"] = "running"
            job["progress"]["coin"] = coin
            now = time.monotonic()
            if phase_changed or total_changed or row.get("completed") == row.get("total") or now - last_download_save >= 0.5:
                last_download_save = now
                save()

    def decide_training(details):
        with lock:
            job.update(status="awaiting_decision", stage="awaiting_decision", pending_decision=dict(details))
            save()
        log(f"Training paused at epoch {details['epoch']}. Continue for 10 more epochs or finish with the best model.")
        while True:
            command = commands.get()
            if (isinstance(command, dict) and set(command) == {"decision_id", "action"}
                    and type(command["decision_id"]) is int
                    and command["decision_id"] == details["decision_id"]
                    and command["action"] in ("continue", "finish")):
                break
        continued = command["action"] == "continue"
        with lock:
            job.update(status="running", stage="training" if continued else "evaluating", pending_decision=None)
            save()
        log("Continuing the same model and optimizer for 10 more epochs." if continued
            else "Finishing training and evaluating the best validation checkpoint.")
        return continued

    try:
        job.update(status="running", stage="starting")
        save()
        if job["operation"] == "forecast":
            from .chart_forecast import forecast_chart
            directory = path.parent.parent / "charts" / job["id"]
            chart = json.loads((directory / "input.json").read_text(encoding="utf-8"))
            job["stage"] = "forecasting"
            save()
            result = forecast_chart(path.parent.parent.parent, job["config"], chart, progress=log)
            write_json(directory / "result.json", result)
            log("Future candles are ready. Original and fine-tuned model files remain unchanged.")
            job.update(status="completed", stage="completed", finished_at=utc_now())
            save()
            return 0
        config = TrainingConfig(**job["config"])
        if job["operation"] == "train":
            log("Loading local candle history. Each training run is saved separately.")
            callbacks = {"on_training_decision": decide_training} if commands is not None else {}
            summary = train(config, progress=log, on_event=event, **callbacks)
            tokenizer_only = summary.get("artifact_type") == "kronos-tokenizer"
            epoch_limit = (config.kronos_tokenizer_epochs if tokenizer_only else
                           summary.get("training_control", {}).get("epoch_limit",
                               job["progress"].get("max_epochs", config.epochs)))
            job["progress"].update(best_epoch=summary["best_epoch"], max_epochs=epoch_limit)
            log("Tokenizer training completed. The saved tokenizer is ready for predictor training."
                if tokenizer_only else "Training and held-out evaluation completed. Results are ready.")
        else:
            # Two coins prepare independently. Only this coordinator publishes,
            # so each four-timeframe replacement still shares the parent's Stop lock.
            download_dir = path.parent / f"{job['id']}-data"
            download_dir.mkdir(parents=True, exist_ok=True)
            job["stage"] = "downloading"
            job["progress"].update(total_coins=len(config.coins), completed_coins=0,
                                   coin_progress={coin: {"status": "queued", "phase": "queued", "completed": 0,
                                                        "total": None} for coin in config.coins})
            save()
            now = datetime.now(timezone.utc)
            cutoff = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0).isoformat()
            end = config.end or cutoff
            result = {}

            def prepare_coin(index, coin):
                log(f"Checking saved {coin} candles ({index}/{len(config.coins)}), then fetching missing history.")
                download_progress({"coin": coin, "phase": "checking_local", "completed": None, "total": None})
                entry = download_multiframe_history([coin], config.start, end, download_dir,
                                                    archive_root=Path(config.data_dir) / "_archives",
                                                    existing_data_dir=config.data_dir,
                                                    on_progress=download_progress)[coin]
                if entry["status"] == "failed":
                    raise ValueError("; ".join(entry.get("errors", [])) or "History download failed.")
                if config.end is None:
                    entry["recent_refresh"] = refresh_multiframe_recent(
                        download_dir, [coin], as_of=cutoff, on_progress=download_progress)[coin]
                metadata = {interval: read_json(download_dir / interval / f"{coin}.metadata.json")
                            for interval in ("15m", "1h", "4h", "1d")}
                entry.update(timeframes=metadata, rows=metadata["15m"]["rows"],
                             status="partial" if any(meta["status"] == "partial" for meta in metadata.values()) else "downloaded")
                return entry

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {executor.submit(prepare_coin, index, coin): coin
                           for index, coin in enumerate(config.coins, 1)}
                for future in as_completed(futures):
                    coin = futures[future]
                    row = job["progress"]["coin_progress"][coin]
                    try:
                        result[coin] = future.result()
                        write_json(download_dir / "download_report.json", result)
                        # Freeze progress writers through the acknowledgement.
                        # The parent updates the same snapshot while publishing;
                        # fetchers must not restore its previous Ready state.
                        with lock:
                            try:
                                row.update(status="ready", rows=result[coin]["rows"], source=result[coin].get("source"),
                                           phase="prepared", completed=4, total=4)
                                job.update(stage="publishing_coin", publishing_coin=coin)
                                save()
                                if commands is None:
                                    publish_download(download_dir, config.data_dir, [coin])
                                else:
                                    while True:
                                        response = commands.get()
                                        if isinstance(response, dict) and response.get("coin") == coin and type(response.get("published")) is bool:
                                            break
                                    if not response["published"]:
                                        raise ValueError(response.get("error", "Could not save downloaded candles."))
                                row.update(status="saved")
                            except Exception as error:
                                row.update(status="failed", error=str(error))
                                raise
                            finally:
                                job["progress"]["completed_coins"] = sum(
                                    entry["status"] == "saved" for entry in job["progress"]["coin_progress"].values())
                                job.update(stage="downloading", publishing_coin=None)
                                save()
                        log(f"{coin}: saved {row['rows']:,} native 15m candles and all four timeframes. "
                            f"{result[coin].get('reused_rows', 0):,} candles reused from saved history.")
                    except Exception as error:
                        with lock:
                            row.update(status="failed", error=str(error))
                            save()
                        result.setdefault(coin, {"rows": 0})
                        result[coin].update(status="failed", errors=[str(error)])
                        write_json(download_dir / "download_report.json", result)
                        log(f"{coin}: failed: {error}. Previously saved coins remain available.")
            failed = [coin for coin in config.coins if result[coin]["status"] == "failed"]
            if failed:
                raise ValueError(f"History download failed for {', '.join(failed)}. Download diagnostics: {download_dir / 'download_report.json'}")
            log("Candle history is ready. Every completed coin has been saved.")
        job.update(status="completed", stage="completed", pending_decision=None, finished_at=utc_now())
        save()
        return 0
    except Exception as error:
        traceback.print_exc()
        if job.get("operation") == "download":
            for row in job.get("progress", {}).get("coin_progress", {}).values():
                if row.get("status") == "running":
                    row.update(status="failed", error=str(error))
        job.update(status="failed", stage="failed", pending_decision=None, error=str(error), finished_at=utc_now())
        log(f"Job failed: {error}")
        return 1


def _read_parent_block(fd, size):
    if os.name != "nt":
        return os.read(fd, size)

    import _winapi
    import msvcrt

    handle = msvcrt.get_osfhandle(fd)
    while True:
        try:
            available = _winapi.PeekNamedPipe(handle)[0]
        except BrokenPipeError:
            return b""
        if available:
            return os.read(fd, min(size, available))
        # An outstanding synchronous stdin read can block DuplicateHandle in
        # Windows-spawned preparation children. Never read an empty command
        # pipe; peeking also detects parent EOF without holding a pending read.
        time.sleep(0.05)


def read_parent_commands(commands, fd=None):
    """Read bounded JSON lines, while retaining parent-EOF cancellation."""
    fd = sys.stdin.fileno() if fd is None else fd
    pending = b""
    while True:
        try:
            block = _read_parent_block(fd, 1024)
        except OSError:
            os._exit(3)
        if not block:
            # The dashboard owns the write side. A crashed parent closes it,
            # so its worker cannot continue training or waiting invisibly.
            os._exit(3)
        pending += block
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            if len(line) > 1024:
                os._exit(3)
            try:
                command = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            training = (isinstance(command, dict) and set(command) == {"decision_id", "action"}
                    and type(command["decision_id"]) is int and command["decision_id"] > 0
                    and command["action"] in ("continue", "finish"))
            publication = (isinstance(command, dict) and set(command) in ({"coin", "published"}, {"coin", "published", "error"})
                           and isinstance(command["coin"], str) and command["coin"].isalnum()
                           and len(command["coin"]) <= 20 and type(command["published"]) is bool
                           and isinstance(command.get("error", ""), str))
            if training or publication:
                try:
                    commands.put_nowait(command)
                except queue.Full:
                    os._exit(3)
        if len(pending) > 1024:
            os._exit(3)


if __name__ == "__main__":
    # An unbuffered OS reader avoids Python's stdin lock during shutdown.
    # This single reader handles both decisions and parent disappearance.
    commands = queue.Queue(maxsize=1)
    threading.Thread(target=read_parent_commands, args=(commands,), daemon=True).start()
    raise SystemExit(run_job(sys.argv[1], commands=commands))
