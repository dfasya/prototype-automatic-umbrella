"""Extract held-out clips from labeled_dataset/ to replay to the umbrella later.

Rebuilds the train/evaluation split of training.ipynb (``hourly_split`` with the
same settings), so every extracted row is one that no model was trained on: an
evaluation hour or the buffer around it. A clip is one unbroken stretch of such
rows, kept whole with its real 20 s timestamps and labels:

    lead_in (30 min buffer) -> eval hour(s) -> tail (30 min buffer)

The lead-in gives the 15-minute feature windows real history before the first
scored row, as the device would have, and the tail holds the labels up to 30
minutes past the last evaluation row (the model predicts 5 minutes ahead).

Clips are sorted into rain, heat, heat+rain (the main reason they are OPEN) and
dry (never OPEN). Within each category, clips where the umbrella both opens and
closes are picked first, then clips with only one of the two, and the picks are
spread across months so every season is represented.

    python deployment/extract_evaluation.py                 # default selection
    python deployment/extract_evaluation.py --rain 40 --dry 0
    python deployment/extract_evaluation.py --all           # every held-out clip

Writes ``clips.csv`` (one row per 20 s step, ``clip_id`` per clip) and
``manifest.csv`` (one row per clip) to ``--out`` (default deployment/evaluation_dataset/).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# This script needs the training code in REPO/modules. Python would otherwise
# import deployment/modules (the runtime package, which has an __init__.py and
# so wins over the namespace package REPO/modules), so drop this folder first.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.path.insert(0, str(REPO))

from modules.dataset.split import hourly_split  # noqa: E402
from modules.dataset.windowing import MAX_WINDOW  # noqa: E402

# Device sensors (no barometer, no rain gauge), UTCI, and the labels.
COLUMNS = ["temperature", "humidity", "wind_speed", "wind_dir", "rain_flag",
           "utci", "p_open", "label", "reason"]
CATEGORIES = ["rain", "heat", "heat+rain", "dry"]


def load_labeled(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSVs in {directory}: run labeling.ipynb first")
    return pd.concat(
        pd.read_csv(f, index_col="timestamp", parse_dates=["timestamp"],
                    usecols=["timestamp"] + COLUMNS, keep_default_na=False)
        for f in files
    ).sort_index()


def find_clips(data: pd.DataFrame, step: pd.Timedelta, horizon: pd.Timedelta) -> pd.DataFrame:
    """Every unbroken stretch of non-training rows, with its phases and a summary."""
    held_out = (data["split"] != "train").to_numpy()
    t = data.index.asi8
    starts = np.r_[True, (np.diff(t) != step.value) | (held_out[1:] != held_out[:-1])]
    run = np.cumsum(starts)

    clips, rows = [], []
    for _, part in data[held_out].groupby(run[held_out]):
        is_eval = (part["split"] == "eval").to_numpy()
        if not is_eval.any():
            continue
        first, last = np.flatnonzero(is_eval)[[0, -1]]
        lead_in = first * step
        if lead_in < MAX_WINDOW:  # windows would still be filling at the first scored row
            continue
        phase = np.where(is_eval, "eval", "gap").astype(object)
        phase[:first], phase[last + 1:] = "lead_in", "tail"

        is_open = (part["label"] == "OPEN").to_numpy().astype(int)
        moves = np.diff(is_open)
        reasons = part.loc[part["label"] == "OPEN", "reason"]
        category = reasons.value_counts().idxmax() if len(reasons) else "dry"
        start = part.index[0]
        clip_id = f"{start:%Y%m%d_%H%M}"

        future = (part["label"] == "OPEN").astype(float).reindex(part.index + horizon)
        rows.append(part[COLUMNS].rename(columns={"p_open": "label_p_open"}).assign(
            clip_id=clip_id, phase=phase,
            label_ahead=np.where(future.isna(), "", np.where(future == 1, "OPEN", "CLOSED")),
        ))
        clips.append({
            "clip_id": clip_id, "start": start, "end": part.index[-1] + step,
            "minutes": len(part) * step / pd.Timedelta(minutes=1),
            "lead_in_min": lead_in / pd.Timedelta(minutes=1),
            "eval_min": is_eval.sum() * step / pd.Timedelta(minutes=1),
            "category": category,
            "open_share": round(is_open.mean(), 3),
            "openings": int((moves == 1).sum()),
            "closings": int((moves == -1).sum()),
            "max_utci": round(part["utci"].max(), 1),
            "wet_share": round((part["rain_flag"] > 0).mean(), 3),
        })
    manifest = pd.DataFrame(clips).set_index("clip_id")
    manifest["full_cycle"] = (manifest["openings"] > 0) & (manifest["closings"] > 0)
    return manifest, rows


def select(manifest: pd.DataFrame, quotas: dict[str, int], seed: int) -> pd.Index:
    """Up to ``quotas[c]`` clips per category: full open/close cycles first, then
    clips with one move, then the rest, round-robin across months."""
    rng = np.random.default_rng(seed)
    moves = (manifest["openings"] > 0).astype(int) + (manifest["closings"] > 0).astype(int)
    month = manifest["start"].dt.to_period("M")
    chosen = []
    for category, quota in quotas.items():
        picked = []
        for level in (2, 1, 0):
            pool = manifest.index[(manifest["category"] == category) & (moves == level)]
            by_month = [list(rng.permutation(ids)) for _, ids in
                        pd.Series(pool, index=pool).groupby(month[pool])]
            while by_month and len(picked) < quota:
                for ids in by_month:
                    if ids and len(picked) < quota:
                        picked.append(ids.pop())
                by_month = [ids for ids in by_month if ids]
        chosen += picked
    return manifest.loc[chosen].sort_values("start").index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--labeled", type=Path, default=REPO / "labeled_dataset",
                        help="labeled 20 s data from labeling.ipynb (default: labeled_dataset/)")
    parser.add_argument("--out", type=Path, default=HERE / "evaluation_dataset",
                        help="folder for clips.csv and manifest.csv (default: deployment/evaluation_dataset/)")
    parser.add_argument("--rain", type=int, default=20, help="clips mainly OPEN for rain (default: 20)")
    parser.add_argument("--heat", type=int, default=20, help="clips mainly OPEN for heat (default: 20)")
    parser.add_argument("--mixed", type=int, default=5, help="clips mainly OPEN for heat+rain (default: 5)")
    parser.add_argument("--dry", type=int, default=15, help="clips that stay CLOSED (default: 15)")
    parser.add_argument("--all", action="store_true", default=False,
                        help="every held-out clip instead of the quotas above (default: off)")
    parser.add_argument("--seed", type=int, default=0, help="seed for picking clips (default: 0)")
    split = parser.add_argument_group("split (must match training.ipynb, section 2)")
    split.add_argument("--n-hours", type=int, default=3, help="evaluation hours per day (default: 3)")
    split.add_argument("--split-seed", type=int, default=42, help="split seed (default: 42)")
    split.add_argument("--gap-minutes", type=float, default=30,
                       help="buffer around each evaluation hour (default: 30)")
    split.add_argument("--earliest-hour", type=int, default=6,
                       help="no evaluation hours before this hour (default: 6)")
    split.add_argument("--horizon-minutes", type=float, default=5,
                       help="how far ahead label_ahead looks, the model's HORIZON_MINUTES (default: 5)")
    args = parser.parse_args()

    data = load_labeled(args.labeled)
    step = data.index.to_series().diff().median()
    data["split"], _ = hourly_split(data, n_hours=args.n_hours, seed=args.split_seed,
                                    gap_minutes=args.gap_minutes, stratified=True,
                                    earliest_hour=args.earliest_hour)
    manifest, rows = find_clips(data, step, pd.Timedelta(minutes=args.horizon_minutes))
    print(f"{len(data):,} rows, {len(manifest):,} held-out clips available:")
    print(manifest.groupby("category")["full_cycle"].agg(clips="size", full_cycle="sum")
          .reindex(CATEGORIES).fillna(0).astype(int).to_string())

    if args.all:
        keep = manifest.sort_values("start").index
    else:
        quotas = dict(zip(CATEGORIES, [args.rain, args.heat, args.mixed, args.dry]))
        keep = select(manifest, quotas, args.seed)
    by_id = {r["clip_id"].iloc[0]: r for r in rows}
    clips = pd.concat([by_id[c] for c in keep])
    clips = clips[["clip_id", "phase"] + [c for c in clips if c not in ("clip_id", "phase")]]

    save_together({args.out / "clips.csv": clips, args.out / "manifest.csv": manifest.loc[keep]})
    chosen = manifest.loc[keep]
    print(f"\nwrote {len(keep)} clips, {len(clips):,} rows "
          f"({chosen['minutes'].sum() / 60:.1f} h, {chosen['eval_min'].sum() / 60:.1f} h scored) to {args.out}")
    print(chosen.groupby("category").agg(clips=("minutes", "size"), full_cycle=("full_cycle", "sum"),
                                         openings=("openings", "sum"), closings=("closings", "sum"),
                                         months=("start", lambda s: s.dt.to_period("M").nunique()))
          .reindex([c for c in CATEGORIES if c in set(chosen["category"])]).to_string())


def save_together(frames: dict[Path, pd.DataFrame]) -> None:
    """Write several CSVs so that an interruption never leaves a half-written or mismatched set.

    Every file is first written in full to a temporary name next to its target;
    only then are they renamed into place, one quick rename each. Ctrl+C before
    that keeps all existing files unchanged.
    """
    tmps = {path: path.with_name(f".{path.name}.tmp") for path in frames}
    try:
        for path, frame in frames.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(tmps[path])
        for path, tmp in tmps.items():
            tmp.replace(path)
    finally:
        for tmp in tmps.values():
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped; existing clips.csv / manifest.csv were left unchanged")
        sys.exit(130)
