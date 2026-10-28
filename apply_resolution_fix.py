"""Apply the timestamp-resolution fix to this checkout.

Run it once from the repository root:

    python apply_resolution_fix.py

It edits six places where the code compared ``DatetimeIndex.asi8`` against a
nanosecond constant. ``asi8`` returns the index's own time unit, which is
microseconds on pandas 2.0 and later when timestamps are parsed from strings,
while ``Timedelta.value`` is always nanoseconds. Where the two disagree the
comparison is wrong by a factor of 1000 and nothing raises: segment boundaries
land on every row, the controller resets at every reading, and the feature
windows never fill.

The fix is one call per site, ``.as_unit("ns")`` before ``.asi8``, which pins
the comparison to nanoseconds on every pandas version.

Nothing is written unless all six patterns match exactly once. Run it a second
time and it reports that the work is already done; it will not double-apply.
Use ``--dry-run`` to see what would change without touching anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# (file, text to find, replacement). Each pattern must occur exactly once.
EDITS: list[tuple[str, str, str]] = [
    (
        "modules/model/hmm.py",
        "    breaks = np.r_[False, np.diff(predictions.index.asi8) != step.value]",
        '    breaks = np.r_[False, np.diff(predictions.index.as_unit("ns").asi8) != step.value]',
    ),
    (
        "modules/dataset/dataset.py",
        "    step_ok = np.r_[False, np.diff(ts.asi8) == 60 * 10**9]",
        '    step_ok = np.r_[False, np.diff(ts.as_unit("ns").asi8) == 60 * 10**9]',
    ),
    (
        "modules/dataset/split.py",
        "        t = df.index.asi8\n        starts = eval_starts.asi8",
        '        t = df.index.as_unit("ns").asi8\n'
        '        starts = eval_starts.as_unit("ns").asi8',
    ),
    (
        "modules/dataset/split.py",
        "    breaks = np.flatnonzero(np.diff(index.asi8) != step.value) + 1",
        '    breaks = np.flatnonzero(np.diff(index.as_unit("ns").asi8) != step.value) + 1',
    ),
    (
        "deployment/modules/dataset.py",
        "    breaks = np.r_[True, np.diff(index.asi8) != step.value]",
        '    breaks = np.r_[True, np.diff(index.as_unit("ns").asi8) != step.value]',
    ),
    (
        "deployment/extract_evaluation.py",
        "    t = data.index.asi8",
        '    t = data.index.as_unit("ns").asi8',
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change and write nothing")
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="repository root (default: the current directory)")
    args = parser.parse_args()
    root = args.root.resolve()

    if not (root / "deployment").is_dir() or not (root / "modules").is_dir():
        print(f"{root} does not look like the repository root "
              "(no 'modules' and 'deployment' directories here).")
        print("cd into the repository first, then run this again.")
        return 1

    # Read every file once, so a failure anywhere leaves the checkout untouched.
    texts: dict[str, str] = {}
    pending: list[tuple[str, str, str]] = []
    already = 0
    problems: list[str] = []

    for rel, old, new in EDITS:
        path = root / rel
        if not path.exists():
            problems.append(f"{rel}: file not found")
            continue
        text = texts.setdefault(rel, path.read_text(encoding="utf-8"))
        staged = "".join(t for r, t in ((r, t) for r, _, t in pending) if r == rel)
        if old in text:
            if text.count(old) != 1:
                problems.append(f"{rel}: found {text.count(old)} copies of a pattern, expected 1")
                continue
            pending.append((rel, old, new))
        elif new in text or new in staged:
            already += 1
        else:
            problems.append(f"{rel}: pattern not found — this checkout differs from the "
                            "version the fix was written against")

    if problems:
        print("Nothing was changed. Problems found:\n")
        for line in problems:
            print(f"  {line}")
        print("\nSend this output back before editing anything by hand.")
        return 1

    if not pending:
        print(f"All {already} sites are already fixed. Nothing to do.")
        return 0

    print(f"{len(pending)} site(s) to fix"
          + (f", {already} already done" if already else "") + ":\n")
    for rel, old, _ in pending:
        first = old.strip().splitlines()[0]
        print(f"  {rel}\n      {first}")

    if args.dry_run:
        print("\n--dry-run: nothing was written.")
        return 0

    for rel, old, new in pending:
        texts[rel] = texts[rel].replace(old, new, 1)
    for rel, text in texts.items():
        (root / rel).write_text(text, encoding="utf-8")

    print(f"\nWritten. {len({r for r, _, _ in pending})} file(s) changed.")
    print("Next: cd deployment && python reproducibility_check.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
