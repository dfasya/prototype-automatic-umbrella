"""Check that this machine reproduces the reported deployment results.

Run it from the ``deployment`` directory:

    python reproducibility_check.py

It prints the library versions, the timestamp resolution pandas chose on this
machine, whether the resolution patch is present in the source, the segment
count, the metric table, and a fingerprint of the predictions. Two machines that
print the same fingerprint produced the same decisions, reading for reading.

Why this exists
---------------
Several places in this codebase compared ``DatetimeIndex.asi8`` against
``Timedelta.value``. ``asi8`` returns the index's own time unit, which is
microseconds on pandas 2.0 and later when timestamps are parsed from strings;
``Timedelta.value`` is always nanoseconds. Where the two units disagree the
comparison is wrong by a factor of 1000, every reading looks like the start of a
new segment, and the runtime resets the model at every 20-second step so the
feature windows never fill. Nothing raises, and the metrics simply come out
worse. This script makes that condition visible instead of silent.
"""
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EXPECTED_CLIPS = 60
PATCH_MARK = 'as_unit("ns").asi8'

# Files that carried the comparison, relative to the repository root.
PATCH_SITES = [
    "deployment/modules/dataset.py",
    "deployment/extract_evaluation.py",
    "modules/dataset/dataset.py",
    "modules/dataset/split.py",
    "modules/model/hmm.py",
]


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def patch_status(root: Path) -> list[tuple[str, str]]:
    """For each known site: patched, unpatched, or missing from this checkout."""
    rows = []
    for rel in PATCH_SITES:
        path = root / rel
        if not path.exists():
            rows.append((rel, "file not found"))
            continue
        text = path.read_text(encoding="utf-8")
        if PATCH_MARK in text:
            rows.append((rel, "patched"))
        elif ".asi8" in text:
            rows.append((rel, "NOT PATCHED"))
        else:
            rows.append((rel, "no comparison here"))
    return rows


def fingerprint(predictions: pd.DataFrame) -> str:
    """A hash that ignores float formatting but not the decisions themselves."""
    parts = [
        np.round(predictions["p_open"].to_numpy(dtype=float), 9).tobytes(),
        predictions["open"].to_numpy().astype(np.int8).tobytes(),
        predictions["state"].to_numpy().astype(np.int16).tobytes(),
    ]
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()[:16]


def main() -> None:
    root = HERE.parent

    rule("environment")
    print(f"python  {platform.python_version()}  ({platform.system()} {platform.machine()})")
    print(f"pandas  {pd.__version__}")
    print(f"numpy   {np.__version__}")

    rule("timestamp resolution on this machine")
    sample = pd.to_datetime(
        ["2024-01-01 00:00:00", "2024-01-01 00:00:20"], format="ISO8601"
    )
    step = pd.Timedelta(seconds=20)
    raw = int(np.diff(sample.asi8)[0])
    fixed = int(np.diff(sample.as_unit("ns").asi8)[0])
    print(f"parsed unit                     : {sample.unit}")
    print(f"np.diff(index.asi8)             : {raw:,}")
    print(f"np.diff(index.as_unit('ns').asi8): {fixed:,}")
    print(f"Timedelta(seconds=20).value     : {step.value:,}")
    if raw == step.value:
        print("-> units agree here; the unpatched comparison happens to work on this machine")
    else:
        print(f"-> units DISAGREE by a factor of {step.value // raw}; "
              "the unpatched comparison is wrong on this machine")

    rule("patch status")
    for rel, status in patch_status(root):
        print(f"{status:>20}  {rel}")

    rule("segments")
    from modules import dataset as ds

    clips = ds.load_clips()
    segments = int(clips["segment"].nunique())
    print(f"clips in the evaluation set : {clips['clip_id'].nunique()}")
    print(f"segments the loader found   : {segments}   (expect {EXPECTED_CLIPS}, one per clip)")
    if segments != EXPECTED_CLIPS:
        print("-> WRONG. Every reading is being treated as its own segment; "
              "the model is reset at each 20-second step and never warms up.")

    from modules.inference import UmbrellaModel, _moves_per_day, run_stream, summarize

    predictions = run_stream(UmbrellaModel.load(), clips)
    restarts = int(predictions["restarted"].sum())
    print(f"restarts during the replay  : {restarts}   (expect {EXPECTED_CLIPS})")

    rule("metrics")
    pd.set_option("display.width", 120)
    columns = ["rows", "open_share", "precision", "recall", "f1", "false_open"]
    print(summarize(predictions, ds.load_manifest())[columns].round(3).to_string())
    print(f"\nmoves per 24 h: model {_moves_per_day(predictions, 'open'):.1f}, "
          f"label {_moves_per_day(predictions, 'label'):.1f}")

    rule("fingerprint")
    print(f"predictions fingerprint : {fingerprint(predictions)}")
    print("Two machines printing the same fingerprint produced identical decisions.")


if __name__ == "__main__":
    main()
