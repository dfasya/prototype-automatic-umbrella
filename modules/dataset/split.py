"""Hold out random hours of every day for evaluation.

Each day contributes ``n_hours`` evaluation hours; everything else is training,
except an optional buffer around each evaluation hour. Neighbouring minutes of
weather are nearly identical and the labels use rolling windows, so without a
buffer the training data leaks into the evaluation hours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["hourly_split", "sequence_lengths"]


def _pick_hours(
    available: np.ndarray, n_hours: int, stratified: bool, earliest_hour: int,
    rng: np.random.Generator,
) -> list[int]:
    available = available[available >= earliest_hour]
    if not stratified:
        k = min(n_hours, len(available))
        return sorted(rng.choice(available, size=k, replace=False).tolist())
    picks = []
    for block in np.array_split(np.arange(earliest_hour, 24), n_hours):
        candidates = np.intersect1d(block, available)
        if len(candidates):
            picks.append(int(rng.choice(candidates)))
    return picks


def hourly_split(
    df: pd.DataFrame,
    n_hours: int = 3,
    seed: int = 42,
    gap_minutes: float = 30.0,
    stratified: bool = True,
    earliest_hour: int = 0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Assign every row to "train", "eval" or "gap".

    For each calendar day, ``n_hours`` clock hours that contain data become
    "eval", drawn only from ``earliest_hour`` to 23 (e.g. 6 keeps the night
    and early morning for training). With ``stratified``, that window is cut
    into ``n_hours`` equal blocks (06-11, 12-17, 18-23 for 3 from 06:00) and one
    hour is drawn from each, so the evaluation spreads across the day. Rows
    within ``gap_minutes`` before or after an evaluation hour are "gap" and
    belong to neither set.

    Returns the per-row split labels and a table of the chosen hours.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex.")
    if not 0 <= earliest_hour <= 23:
        raise ValueError("earliest_hour must be between 0 and 23.")
    if not 1 <= n_hours <= 24 - earliest_hour:
        raise ValueError(f"n_hours must be between 1 and {24 - earliest_hour}.")
    rng = np.random.default_rng(seed)

    hour_starts = df.index.floor("h")
    available = pd.Series(hour_starts.unique())
    chosen = []
    for day, hours in available.groupby(available.dt.normalize()):
        picks = _pick_hours(hours.dt.hour.to_numpy(), n_hours, stratified, earliest_hour, rng)
        chosen.extend(day + pd.Timedelta(hours=h) for h in picks)
    eval_starts = pd.DatetimeIndex(sorted(chosen))

    is_eval = hour_starts.isin(eval_starts)
    split = np.where(is_eval, "eval", "train").astype(object)

    if gap_minutes > 0 and len(eval_starts):
        gap = pd.Timedelta(minutes=gap_minutes)
        t = df.index.as_unit("ns").asi8
        starts = eval_starts.as_unit("ns").asi8
        hour = pd.Timedelta(hours=1).value
        # Nearest evaluation hour starting at or before t, and the next one after t.
        prev_i = np.searchsorted(starts, t, side="right") - 1
        next_i = prev_i + 1
        prev_end = np.where(prev_i >= 0, starts[np.clip(prev_i, 0, None)] + hour, np.iinfo(np.int64).min)
        next_start = np.where(next_i < len(starts),
                              starts[np.clip(next_i, None, len(starts) - 1)], np.iinfo(np.int64).max)
        after = (t >= prev_end) & (t < prev_end + gap.value)
        before = (t < next_start) & (t >= next_start - gap.value)
        split[(after | before) & ~is_eval] = "gap"

    hours = pd.DataFrame({"start": eval_starts, "hour": eval_starts.hour})
    hours["day"] = eval_starts.normalize()
    return pd.Series(split, index=df.index, name="split"), hours


def sequence_lengths(index: pd.DatetimeIndex, step: pd.Timedelta) -> np.ndarray:
    """Lengths of the unbroken runs in ``index`` (hmmlearn's ``lengths``).

    A new run starts wherever consecutive timestamps are not exactly ``step``
    apart, e.g. across a data gap or a held-out evaluation hour.
    """
    if len(index) == 0:
        return np.array([], dtype=int)
    breaks = np.flatnonzero(np.diff(index.as_unit("ns").asi8) != step.value) + 1
    return np.diff(np.r_[0, breaks, len(index)])
