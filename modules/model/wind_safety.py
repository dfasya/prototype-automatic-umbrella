"""Wind override for the canopy decision layer.

This sits *after* the model and after ``apply_hysteresis``. It can force the
canopy CLOSED, never open. Nothing in ``modules/model/hmm.py`` has to change:
the override takes the decisions that function already produces and clears the
ones that would leave the fabric extended in damaging wind.

Why it is needed
----------------
In the current pipeline wind reaches the decision only through UTCI, where it
*cools*. Higher wind lowers ``p_heat``, so wind makes a heat opening less
likely. For rain it does nothing at all: ``p_rain`` is 1 whenever the detector
is wet, whatever the wind is doing. A thunderstorm is wet and windy at the same
time, so the present logic extends the canopy into the exact conditions that
damage it.

Thresholds
----------
EN 13561 (external blinds and awnings) sets wind resistance classes. The sensor
setting values a manufacturer publishes against those classes are:

    class 1   7.8 m/s   (28 km/h)
    class 2  10.6 m/s   (38 km/h)
    class 3  13.3 m/s   (48 km/h)

The default here is class 1, the most conservative, because the wind class of
the purchased canopy is not known yet. Raise it once the manufacturer's rating
is in hand. Do not raise it on the grounds that nothing has broken so far.

For scale: the KMA strong-wind advisory for land is 14 m/s sustained. That is
above every class in the table. By the time a KMA advisory exists, a class 3
awning is already past its limit, so a weather warning is not a usable trigger
for this device. The threshold has to be local and much lower.

Two further limits worth stating plainly:

* Wind at AWS station 939 is not wind at the canopy. Terrain, buildings and
  height all change it. A local anemometer is the only honest input here; the
  station series is a stand-in for development.
* A sensor cannot catch a gust that has already arrived. The lockout below
  reduces re-opening into a gust train but does not remove the risk.

Usage
-----
    from modules.model.hmm import apply_hysteresis
    from modules.model.wind_safety import apply_wind_override

    raw = apply_hysteresis(predictions, margin=0.2, min_hold_minutes=3)
    safe, blocked = apply_wind_override(
        raw, predictions.index, stream["wind_max_5m"].to_numpy()
    )

``blocked`` is the mask of rows the override closed, so the cost of the rule
can be measured instead of assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "EN13561_SETTING_MS", "CLOSE_ABOVE_MS", "REOPEN_BELOW_MS", "LOCKOUT_MINUTES",
    "apply_wind_override", "override_summary",
]

# EN 13561 wind resistance classes, manufacturer sensor setting values (m/s).
EN13561_SETTING_MS = {1: 7.8, 2: 10.6, 3: 13.3}

# Defaults: class 1 until the canopy's actual class is known.
CLOSE_ABOVE_MS = EN13561_SETTING_MS[1]
REOPEN_BELOW_MS = 5.5          # hysteresis, must stay below CLOSE_ABOVE_MS
LOCKOUT_MINUTES = 10.0         # minimum closed time after a wind trigger


def _segment_starts(index: pd.DatetimeIndex, step: pd.Timedelta) -> np.ndarray:
    """True where a row does not follow the previous one by exactly ``step``.

    The comparison is between Timedelta objects, not between raw integers.
    ``DatetimeIndex.asi8`` returns the index's own resolution, which is not
    always nanoseconds; comparing it against ``step.value`` silently marks
    every row as a segment start when the two units disagree.
    """
    if len(index) == 0:
        return np.zeros(0, dtype=bool)
    # The first diff is NaT, which compares unequal, so row 0 is a start.
    return (index.to_series().diff() != step).to_numpy()


def _run_override(
    gust: np.ndarray, start: np.ndarray,
    close_above: float, reopen_below: float, lockout_steps: int,
) -> np.ndarray:
    """Boolean mask: True where wind holds the canopy closed."""
    locked = np.zeros(len(gust), dtype=bool)
    state = False
    since_trigger = 0
    for i in range(len(gust)):
        if start[i]:
            state = gust[i] >= close_above
            since_trigger = 0
        elif gust[i] >= close_above:
            state = True
            since_trigger = 0
        elif state and since_trigger >= lockout_steps and gust[i] < reopen_below:
            state = False
        since_trigger += 1
        locked[i] = state
    return locked


def apply_wind_override(
    decisions: np.ndarray,
    index: pd.DatetimeIndex,
    gust: np.ndarray,
    close_above: float = CLOSE_ABOVE_MS,
    reopen_below: float = REOPEN_BELOW_MS,
    lockout_minutes: float = LOCKOUT_MINUTES,
    step: pd.Timedelta | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Force CLOSED while wind is dangerous. Returns ``(decisions, blocked)``.

    ``gust`` should be a gust measure, not a mean: ``wind_max_5m`` rather than
    ``wind_mean_5m``. Averaging hides the peaks that do the damage.

    The override latches at ``close_above`` and releases only once the gust has
    stayed below ``reopen_below`` and ``lockout_minutes`` have passed since the
    last trigger, so a gust train does not produce repeated open/close cycles.
    Each unbroken stretch of the index is handled on its own; a data gap starts
    a fresh state rather than carrying the old one across it.

    ``blocked`` marks the rows the override changed from OPEN to CLOSED. Rows
    that were already CLOSED are not counted, so the mask is the real cost of
    the rule.
    """
    decisions = np.asarray(decisions, dtype=bool)
    gust = np.asarray(gust, dtype=float)
    index = pd.DatetimeIndex(index)
    if not (len(decisions) == len(gust) == len(index)):
        raise ValueError("decisions, index and gust must be the same length.")
    if reopen_below >= close_above:
        raise ValueError("reopen_below must be below close_above.")
    if not np.isfinite(gust).all():
        raise ValueError("gust contains NaN or infinite values.")
    if len(index) == 0:
        return decisions.copy(), np.zeros(0, dtype=bool)

    if step is None:
        step = index.to_series().diff().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError("Could not determine a positive time step.")
    lockout_steps = int(round(pd.Timedelta(minutes=lockout_minutes) / step))

    locked = _run_override(
        gust, _segment_starts(index, step), close_above, reopen_below, lockout_steps)
    out = decisions & ~locked
    return out, decisions & locked


def override_summary(
    before: np.ndarray, after: np.ndarray, blocked: np.ndarray,
    index: pd.DatetimeIndex, step: pd.Timedelta | None = None,
) -> dict:
    """What the rule cost and how often it fired."""
    index = pd.DatetimeIndex(index)
    if step is None:
        step = index.to_series().diff().median()
    minutes = step / pd.Timedelta(minutes=1)
    hours = len(index) * minutes / 60.0
    events = int(np.count_nonzero(np.diff(np.r_[False, blocked].astype(int)) == 1))

    def switches(state: np.ndarray) -> int:
        return int(np.count_nonzero(np.diff(state.astype(int)) != 0))

    return {
        "rows": len(index),
        "open minutes before": float(before.sum() * minutes),
        "open minutes after": float(after.sum() * minutes),
        "minutes closed by wind": float(blocked.sum() * minutes),
        "share of open time removed": float(blocked.sum() / max(before.sum(), 1)),
        "wind closure events": events,
        "events per 24 h": events / hours * 24 if hours else float("nan"),
        "switches before": switches(before),
        "switches after": switches(after),
    }


if __name__ == "__main__":
    # Synthetic check: a wet, windy hour at 20-second resolution.
    idx = pd.date_range("2026-07-15 14:00", periods=180, freq="20s")
    rng = np.random.default_rng(0)

    # The model wants the canopy open for the whole hour (rain).
    want_open = np.ones(len(idx), dtype=bool)

    # Calm, then a squall with gusts, then calm again.
    gust = np.full(len(idx), 3.0)
    gust[45:75] = 9.5 + rng.normal(0, 0.8, 30)     # squall above class 1
    gust[75:95] = 6.5                              # between reopen and close
    gust[95:105] = 8.4                             # second gust train
    gust = np.clip(gust, 0, None)

    safe, blocked = apply_wind_override(want_open, idx, gust)
    summary = override_summary(want_open, safe, blocked, idx)

    print("wind override self-test")
    print(f"  close above {CLOSE_ABOVE_MS} m/s, reopen below {REOPEN_BELOW_MS} m/s, "
          f"lockout {LOCKOUT_MINUTES:g} min")
    for key, value in summary.items():
        print(f"  {key:<28} {value}")

    first = int(np.argmax(blocked))
    last = len(blocked) - 1 - int(np.argmax(blocked[::-1]))
    print(f"  first closed at             {idx[first]}  (gust {gust[first]:.1f} m/s)")
    print(f"  reopened at                 {idx[last + 1]}  (gust {gust[last + 1]:.1f} m/s)")

    assert not safe[blocked].any(), "a blocked row stayed open"
    assert (safe <= want_open).all(), "the override opened something"
    print("  checks passed")
