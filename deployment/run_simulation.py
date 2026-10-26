"""Replay recorded weather through the umbrella, printing every input and decision.

Loads the model (models/umbrella_hmm.npz) and the held-out clips
(evaluation_dataset/), then feeds each clip to the controller one 20 s
reading at a time, as the device would receive them. For every reading it
prints the sensor inputs, what the model infers (window features, likely
weather state, P(OPEN in 5 min)), the decision, and the label to compare
against. Each clip ends with a summary, and the run with overall scores.

    python run_simulation.py                          # every clip, every reading
    python run_simulation.py --clip 20240709_1130     # one clip (repeatable)
    python run_simulation.py --category heat --limit 3
    python run_simulation.py --every 3 --features     # one line per minute, with features
    python run_simulation.py --speed 60               # 60x real time (one reading per 1/3 s)
    python run_simulation.py --moves-only             # only the lines where the umbrella moves

Ctrl+C stops the replay at once: the clip in progress is dropped, the scores
of the finished clips are printed, and --out is saved with those clips.

Row columns: time, phase (lead_in / eval / gap / tail; only eval rows are
scored), sensors (temperature, humidity, wind, rain detector, UTCI), model
(P(OPEN), most likely state), decision, label now, label 5 min ahead, and
whether the decision matches the label ahead (scored rows only).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from modules.dataset import STEP, load_clips, load_manifest, readings, save_csv  # noqa: E402
from modules.inference import MODEL_PATH, Umbrella, UmbrellaModel, summarize  # noqa: E402


class Style:
    """ANSI colours, or nothing when output is not a terminal or --no-color."""

    def __init__(self, enabled: bool):
        if enabled and os.name == "nt":
            os.system("")  # turn on ANSI escape handling in the Windows console
        codes = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33",
                 "blue": "34", "cyan": "36"}
        for name, code in codes.items():
            setattr(self, name, (lambda c: lambda s: f"\033[{c}m{s}\033[0m")(code) if enabled
                    else (lambda s: s))


def header(s: Style, features: bool) -> str:
    cols = (f"{'time':19}  {'phase':7} | {'temp':>6} {'hum':>5} {'wind':>8} {'rain':4} {'UTCI':>6} |"
            + (f" {'utci10':>6} {'wet5':>4} {'recency':>7} {'dHum15':>6} |" if features else "")
            + f" {'P(OPEN)':>7} {'state':>5} | {'decision':8} | {'label':6} {'in 5m':6} {'':5}")
    return s.bold(cols) + "\n" + s.dim("-" * len(cols))


def row_line(s: Style, t, phase, r, decision, umbrella, label, ahead, features, moved) -> str:
    wet = s.blue("WET ") if r.rain_flag > 0 else "dry "
    sensors = (f"{r.temperature:5.1f}C {r.humidity:4.0f}% {r.wind_speed:4.1f} m/s {wet} "
               f"{r.utci:5.1f}C")
    feats = ""
    if features:
        f = umbrella.last_features
        feats = (f" {f['utci_mean_10m']:5.1f}C {f['wet_share_5m']:4.2f} {f['rain_recency']:7.2f} "
                 f"{f['humidity_chg_15m']:+6.1f} |")
    bar_p = f"{decision.p_open:7.3f}"
    bar_p = s.yellow(bar_p) if decision.p_open >= umbrella.model.threshold - umbrella.model.close_margin else bar_p
    pos = s.green("OPEN    ") if decision.open else "CLOSED  "
    if moved:
        pos = s.bold(s.green("OPENS   ") if decision.open else s.cyan("CLOSES  "))
    scored = phase == "eval" and ahead != ""
    check = ""
    if scored:
        check = s.green(" ok  ") if decision.open == (ahead == "OPEN") else s.red(" MISS")
    phase_txt = phase.ljust(7) if scored else s.dim(phase.ljust(7))
    notes = []
    if decision.restarted:
        notes.append("restart")
    if decision.warming_up:
        notes.append("warming up")
    note = s.dim("  (" + ", ".join(notes) + ")") if notes else ""
    return (f"{t:%Y-%m-%d %H:%M:%S}  {phase_txt} | {sensors} |{feats} {bar_p} {decision.state:5d} | "
            f"{pos} | {label:6} {ahead or '-':6} {check}{note}")


def clip_summary(s: Style, rows: pd.DataFrame) -> str:
    scored = rows[(rows["phase"] == "eval") & (rows["label_ahead"] != "")]
    truth = scored["label_ahead"] == "OPEN"
    pred = scored["open"]
    moves = int((rows["open"].to_numpy()[1:] != rows["open"].to_numpy()[:-1]).sum())
    label_moves = int((rows["label"].to_numpy()[1:] != rows["label"].to_numpy()[:-1]).sum())
    agree = (truth == pred).mean() if len(scored) else np.nan
    tp = int((truth & pred).sum())
    f1 = 2 * tp / max(int(truth.sum() + pred.sum()), 1)
    open_min = rows["open"].sum() * STEP / pd.Timedelta(minutes=1)
    label_min = (rows["label"] == "OPEN").sum() * STEP / pd.Timedelta(minutes=1)
    return (f"  umbrella moved {moves}x (label {label_moves}x), open {open_min:.0f} min "
            f"(label {label_min:.0f} min) | scored rows: {len(scored)}, agree {agree:.1%}"
            + (f", F1 {f1:.3f}" if truth.any() else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=MODEL_PATH,
                        help="model file (default: models/umbrella_hmm.npz)")
    parser.add_argument("--clips", type=Path, default=HERE / "evaluation_dataset" / "clips.csv",
                        help="clips to replay; manifest.csv is read from the same folder "
                             "(default: evaluation_dataset/clips.csv)")
    parser.add_argument("--clip", action="append", default=None,
                        help="only this clip_id, repeatable (default: all clips)")
    parser.add_argument("--category", choices=["rain", "heat", "heat+rain", "dry"], default=None,
                        help="only clips of this category (default: all categories)")
    parser.add_argument("--limit", type=int, default=None, help="at most this many clips (default: no limit)")
    parser.add_argument("--every", type=int, default=1,
                        help="print every Nth reading; moves, restarts and misses are always printed "
                             "(default: 1, every reading)")
    parser.add_argument("--moves-only", action="store_true", default=False,
                        help="print only rows where it moves (default: off)")
    parser.add_argument("--features", action="store_true", default=False,
                        help="also print the main window features (default: off)")
    parser.add_argument("--speed", type=float, default=0.0,
                        help="times real time to pace the replay at, 0 = as fast as possible (default: 0)")
    parser.add_argument("--out", type=Path, default=None,
                        help="save every decision to this CSV (default: not saved)")
    parser.add_argument("--no-color", action="store_true", default=False,
                        help="no colours in the output (default: off, colours in a terminal)")
    args = parser.parse_args()
    if args.every < 1:
        parser.error("--every must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    s = Style(sys.stdout.isatty() and not args.no_color)

    # ------------------------------------------------------------ load --
    print(s.bold("Model"), f"  {args.model}")
    model = UmbrellaModel.load(args.model)
    print(f"  {model.describe()}")
    print(f"  features: {', '.join(model.features)}")
    print(f"  OPEN rate per state: {np.round(model.state_open_rates, 3).tolist()}")

    manifest_path = args.clips.with_name("manifest.csv")
    manifest = load_manifest(manifest_path) if manifest_path.exists() else None
    clip_ids = args.clip
    if manifest is not None and (args.category or args.limit):
        chosen = manifest.index if clip_ids is None else pd.Index(clip_ids)
        if args.category:
            chosen = chosen[manifest.loc[chosen, "category"] == args.category]
        clip_ids = list(chosen[:args.limit] if args.limit else chosen)
        if not clip_ids:
            sys.exit("no clips match the selection")
    clips = load_clips(args.clips, clip_ids)
    ids = list(pd.unique(clips["clip_id"]))
    print(s.bold("\nData"), f"   {args.clips}")
    print(f"  {len(ids)} clips, {len(clips):,} readings at {STEP.total_seconds():.0f} s "
          f"({len(clips) * STEP / pd.Timedelta(hours=1):.1f} h), {clips.index.min():%Y-%m-%d} to "
          f"{clips.index.max():%Y-%m-%d}")
    if manifest is not None:
        counts = manifest.loc[ids, "category"].value_counts()
        print("  categories: " + ", ".join(f"{c} {n}" for c, n in counts.items()))
    pace = STEP.total_seconds() / args.speed if args.speed > 0 else 0.0

    # --------------------------------------------------------- simulate --
    umbrella = Umbrella(model)
    results = []
    try:
        replay(s, args, ids, clips, manifest, umbrella, pace, results)
    except KeyboardInterrupt:
        print("\n" + s.yellow(f"Stopped by user after {len(results)} of {len(ids)} clips; "
                              "the clip in progress is not counted."))
        if not results:
            return

    # ---------------------------------------------------------- results --
    predictions = pd.concat(results)
    print("\n" + s.bold("Overall") + f"  scored rows (phase == eval) vs the label "
          f"{model.horizon_minutes:g} min ahead; persistence = 'the label stays as it is now'")
    table = summarize(predictions, manifest)
    print(table[["rows", "open_share", "precision", "recall", "f1", "false_open"]].round(3).to_string())
    if args.out:
        save_csv(predictions, args.out)
        print(f"\nwrote {len(predictions):,} decisions to {args.out}")


def replay(s, args, ids, clips, manifest, umbrella, pace, results) -> None:
    """Feed every clip to the controller; each finished clip is appended to ``results``."""
    for number, clip_id in enumerate(ids, 1):
        clip = clips[clips["clip_id"] == clip_id]
        info = ""
        if manifest is not None and clip_id in manifest.index:
            m = manifest.loc[clip_id]
            info = (f"  {m['category']}, label opens {m['openings']}x / closes {m['closings']}x, "
                    f"max UTCI {m['max_utci']:.1f}C, wet {m['wet_share']:.0%}")
        print("\n" + s.bold(f"Clip {number}/{len(ids)}  {clip_id}") +
              f"  {clip.index[0]:%Y-%m-%d %H:%M} to {clip.index[-1] + STEP:%H:%M}" + info)
        print(header(s, args.features))

        umbrella.reset()
        rows, previous = [], None
        phases, labels, aheads = (clip[c].to_numpy() for c in ("phase", "label", "label_ahead"))
        for i, r in enumerate(readings(clip)):
            r = r.with_utci()
            decision = umbrella.step(r)
            moved = previous is not None and decision.open != previous and not decision.restarted
            previous = decision.open
            miss = phases[i] == "eval" and aheads[i] != "" and decision.open != (aheads[i] == "OPEN")
            show = moved if args.moves_only else (moved or decision.restarted or i % args.every == 0
                                                  or (miss and args.every > 1))
            if show:
                print(row_line(s, r.timestamp, phases[i], r, decision, umbrella,
                               labels[i], aheads[i], args.features, moved), flush=pace > 0)
            rows.append({"timestamp": r.timestamp, "clip_id": clip_id, "phase": phases[i],
                         "p_open": decision.p_open, "open": decision.open, "state": decision.state,
                         "decision": "OPEN" if decision.open else "CLOSED",
                         "label": labels[i], "label_ahead": aheads[i]})
            if pace:
                time.sleep(pace)
        frame = pd.DataFrame(rows).set_index("timestamp")
        print(clip_summary(s, frame))
        results.append(frame)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # Ctrl+C while loading, before the replay starts
        print("\nstopped, nothing written")
        sys.exit(130)
