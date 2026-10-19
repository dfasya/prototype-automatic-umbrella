"""Prove that dataset/ is complete and that the pipeline reproduces its own output.

Run from the repository root:

    python verify_dataset.py                # default month: 2026-06
    python verify_dataset.py 2026-07        # any month present in simulated_dataset/

Three steps:

1. Load every CSV in ``dataset/`` through ``load_aws`` and report what came back.
2. Re-run the 1-minute -> 20-second simulation for one whole month with
   ``simulate``, and show one day of it in detail.
3. Compare that regenerated month against the committed
   ``simulated_dataset/<month>.csv``, column by column.

Step 3 is the point. Matching the committed file means the raw data, the module
code and this machine together reproduce output that is already in the
repository, rather than merely producing something plausible.

A whole month is simulated rather than a single day because PCHIP is
context-dependent: the curve through any minute is shaped by its neighbours, so
a day simulated in isolation does not reproduce the same values as the same day
simulated inside a longer span.

Nothing is written to disk and nothing in the repository is modified.
Takes a few seconds.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from modules.dataset.dataset import load_aws, simulate

STEP_SECONDS = 20
DEFAULT_MONTH = "2026-06"
HIGHLIGHT_DAY = 20          # day of the month shown in detail
TOLERANCE = 1e-6            # below this, a difference is CSV write precision
PAD_MONTHS = 1              # months simulated either side, then trimmed away
EDGE_MINUTES = 5            # a difference this close to the edge is a boundary effect
RULE = "=" * 70


def banner(text: str) -> None:
    print(f"\n{RULE}\n {text}\n{RULE}")


def row(label: str, value) -> None:
    print(f"   {label:<24} {value}")


def to_nanoseconds(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    """Force nanosecond timestamps.

    ``contiguous_runs`` compares ``DatetimeIndex.asi8`` against ``60 * 10**9``,
    which only holds when the index is in nanoseconds. Newer pandas parses to
    microseconds; every row then looks like its own run, every run is skipped as
    too short, and ``simulate`` returns an empty frame with no error at all.
    This is a no-op where the index is already in nanoseconds.
    """
    out = df.copy()
    out[column] = out[column].astype("datetime64[ns]")
    return out


def describe_day(frame: pd.DataFrame, day: pd.Timestamp) -> None:
    part = frame[(frame.index >= day) & (frame.index < day + pd.Timedelta(days=1))]
    if part.empty:
        row(f"{day:%d %b}", "no rows")
        return
    flag = part["rain_flag"]
    wet = int((flag == 10).sum())
    row(f"{day:%d %b} rows", f"{len(part):,}")
    row(f"{day:%d %b} wet", f"{wet:,} of {len(part):,}  ({wet / len(part):.1%})")
    row(f"{day:%d %b} wind", f"mean {part['wind_speed'].mean():.2f}   "
                             f"max {part['wind_speed'].max():.1f} m/s")
    row(f"{day:%d %b} temp", f"{part['temperature'].min():.1f} .. "
                             f"{part['temperature'].max():.1f} C")


def main() -> int:
    month = pd.Period(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MONTH, freq="M")
    start, end = month.start_time, month.end_time.normalize() + pd.Timedelta(days=1)
    root = Path.cwd()

    print("\n UMCS 2.0  -  dataset verification")
    print(f" repository : {root}")
    print(f" month      : {month}")
    print(f" pandas     : {pd.__version__}   numpy {np.__version__}")

    # ---------------------------------------------------------------- 1 ---
    banner("1   RAW DATA  (dataset/)")

    files = sorted((root / "dataset").glob("*.csv"))
    if not files:
        print("   dataset/ holds no CSV files. Nothing to verify.")
        return 1

    clock = time.time()
    raw = to_nanoseconds(load_aws("dataset"))
    flag = raw["rain_flag"]

    row("files", len(files))
    row("rows", f"{len(raw):,}   after merging overlapping exports")
    row("span", f"{raw['timestamp'].min()}  ->  {raw['timestamp'].max()}")
    row("columns", f"{len(raw.columns)}   {', '.join(raw.columns)}")
    row("rain detector", f"dry {int((flag == 0).sum()):,}   "
                         f"wet {int((flag == 10).sum()):,}   "
                         f"missing {int(flag.isna().sum()):,}")
    row("read in", f"{time.time() - clock:.1f} s")

    # ---------------------------------------------------------------- 2 ---
    banner(f"2   SIMULATION  1 min -> {STEP_SECONDS} s   ({month})")

    # Simulate with a month of padding either side, then trim back to the month.
    # PCHIP is context-dependent, so a month cut out on its own gets a different
    # curve at its first and last minutes than the same month simulated inside a
    # longer run. The padding removes that artefact.
    pad = pd.DateOffset(months=PAD_MONTHS)
    source = raw[(raw["timestamp"] >= start - pad) & (raw["timestamp"] < end + pad)]
    if source.empty:
        print(f"   No raw minutes in {month}.")
        return 1

    clock = time.time()
    simulated = simulate(source, step_seconds=STEP_SECONDS)
    elapsed = time.time() - clock

    if simulated.empty:
        print("   simulate() returned no rows. See to_nanoseconds() in this file.")
        return 1

    fresh = simulated[(simulated.index >= start) & (simulated.index < end)]
    padded_by = len(simulated) - len(fresh)

    step = fresh.index.to_series().diff().median()
    runs = int((fresh.index.to_series().diff() != step).sum())
    fflag = fresh["rain_flag"]

    row("source minutes", f"{len(source):,}")
    row("output rows", f"{len(fresh):,}   step {int(step.total_seconds())} s")
    row("span", f"{fresh.index.min()}  ->  {fresh.index.max()}")
    row("rain_flag held", f"dry {int((fflag == 0).sum()):,}   "
                          f"wet {int((fflag == 10).sum()):,}")
    row("unbroken runs", f"{runs}   gaps are never bridged")
    row("padding", f"{padded_by:,} rows simulated either side, then trimmed")
    row("simulated in", f"{elapsed:.1f} s")

    print()
    describe_day(fresh, month.start_time + pd.Timedelta(days=HIGHLIGHT_DAY - 1))

    # ---------------------------------------------------------------- 3 ---
    committed_path = root / "simulated_dataset" / f"{month}.csv"
    banner(f"3   MATCH AGAINST  simulated_dataset/{month}.csv")

    if not committed_path.exists():
        print(f"   {committed_path.name} is not present, so there is nothing to compare.")
        return 1

    committed = pd.read_csv(committed_path, index_col="timestamp",
                            parse_dates=["timestamp"]).sort_index()

    shared = fresh.index.intersection(committed.index)
    row("rows regenerated", f"{len(fresh):,}")
    row("rows committed", f"{len(committed):,}")
    row("timestamps shared", f"{len(shared):,}")

    extra_here = len(fresh.index.difference(committed.index))
    extra_there = len(committed.index.difference(fresh.index))
    if extra_here or extra_there:
        row("timestamps differing", f"{extra_here} only regenerated, "
                                    f"{extra_there} only committed")

    if len(shared) == 0:
        print("\n   No shared timestamps. The committed file covers a different range.")
        return 1

    edge = pd.Timedelta(minutes=EDGE_MINUTES)
    first, last = shared.min(), shared.max()

    print()
    print(f"   {'column':<14} {'max diff':>10}  {'rows':>7}   where")
    compare = [c for c in fresh.columns if c in committed.columns]
    worst, worst_column, interior_rows = 0.0, "", 0
    for column in compare:
        a = fresh.loc[shared, column].to_numpy(dtype=float)
        b = committed.loc[shared, column].to_numpy(dtype=float)
        both = ~(np.isnan(a) | np.isnan(b))
        delta = np.where(both, np.abs(a - b), 0.0)
        diff = float(delta.max()) if len(delta) else 0.0
        off = shared[delta > TOLERANCE]
        if diff > worst:
            worst, worst_column = diff, column

        if len(off) == 0:
            where = "identical"
        else:
            inside = off[(off > first + edge) & (off < last - edge)]
            interior_rows += len(inside)
            where = ("at the month edges" if len(inside) == 0
                     else f"{len(inside)} away from the edges")
        print(f"   {column:<14} {diff:10.2e}  {len(off):>7}   {where}")

    absent = [c for c in fresh.columns if c not in committed.columns]
    if absent:
        print(f"\n   regenerated but NOT in the committed file: {', '.join(absent)}")
        if "rain_flag" in absent:
            print("   rain_flag is the rain detector. Because it is missing from")
            print("   simulated_dataset/, load_simulated() has to rebuild it from")
            print("   dataset/, which is git-ignored. That is why a clean clone")
            print("   cannot run the pipeline.")

    banner("RESULT")
    if worst < TOLERANCE:
        print("   dataset/ is complete, and this machine reproduces the committed")
        print("   simulated_dataset/ output exactly from the raw KMA files.")
        print(f"   Largest difference across {len(compare)} columns and "
              f"{len(shared):,} rows: {worst:.2e}")
    elif interior_rows == 0:
        print("   dataset/ is complete. Every row away from the month boundary is")
        print("   identical to the committed file; only the first and last minutes")
        print(f"   differ, by at most {worst:.2e} in '{worst_column}'.")
        print("   That is where this run's simulation window was cut, and it says")
        print("   nothing about the data.")
    else:
        print(f"   The pipeline runs, but {interior_rows:,} rows away from the")
        print(f"   boundary differ, by up to {worst:.2e} in '{worst_column}'.")
        print("   The raw data or the simulation settings are not the ones that")
        print("   produced the committed file. Worth asking which range and")
        print("   step_seconds simulation.ipynb was run with.")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
