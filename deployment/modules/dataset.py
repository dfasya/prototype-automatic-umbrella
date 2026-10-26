"""Load and format sensor data for the umbrella.

Two ways in:

- **Recorded clips** (``evaluation_dataset/clips.csv`` from ``extract_evaluation.py``):
  ``load_clips`` validates them and returns one regular 20 s stream per clip.
- **Live readings** from the device: ``Reading.from_dict`` turns one sensor
  sample into a ``Reading``; ``readings`` turns a DataFrame into the same.

UTCI is computed the way the labels were (``labeling.ipynb``): shade (mean
radiant temperature = air temperature), wind clipped to 0.5-17 m/s.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "STEP", "SENSOR_COLUMNS", "DEPLOYMENT", "Reading",
    "compute_utci", "add_utci", "load_clips", "load_manifest", "iter_clips", "readings", "save_csv",
]

DEPLOYMENT = Path(__file__).resolve().parents[1]
STEP = pd.Timedelta(seconds=20)

# Sensors on the device: no barometer and no rain gauge.
SENSOR_COLUMNS = ["temperature", "humidity", "wind_speed", "rain_flag"]
OPTIONAL_COLUMNS = ["wind_dir", "utci"]
LABEL_COLUMNS = ["clip_id", "phase", "label", "reason", "label_p_open", "label_ahead"]


@dataclass(frozen=True)
class Reading:
    """One sensor sample, as the device produces it every 20 s."""

    timestamp: pd.Timestamp
    temperature: float          # air temperature, deg C
    humidity: float             # relative humidity, %
    wind_speed: float           # m/s
    rain_flag: float            # rain detector: 0 = dry, > 0 = wet
    utci: float | None = None   # computed from the other sensors when missing

    @classmethod
    def from_dict(cls, values: Mapping) -> "Reading":
        """Reading from a mapping with ``timestamp`` and the sensor columns."""
        missing = [c for c in ["timestamp"] + SENSOR_COLUMNS if c not in values]
        if missing:
            raise KeyError(f"reading is missing {missing}")
        utci = values.get("utci")
        return cls(
            timestamp=pd.Timestamp(values["timestamp"]),
            temperature=float(values["temperature"]),
            humidity=float(values["humidity"]),
            wind_speed=float(values["wind_speed"]),
            rain_flag=float(values["rain_flag"]),
            utci=None if utci is None or pd.isna(utci) else float(utci),
        )

    def with_utci(self) -> "Reading":
        if self.utci is not None:
            return self
        utci = float(compute_utci(self.temperature, self.humidity, self.wind_speed))
        return Reading(self.timestamp, self.temperature, self.humidity,
                       self.wind_speed, self.rain_flag, utci)


def compute_utci(temperature, humidity, wind_speed) -> np.ndarray:
    """UTCI (deg C) in the shade, as used for the labels and the model features."""
    from pythermalcomfort.models import utci

    tdb = np.asarray(temperature, dtype=float)
    result = utci(
        tdb=tdb, tr=tdb,
        v=np.clip(np.asarray(wind_speed, dtype=float), 0.5, 17.0),
        rh=np.clip(np.asarray(humidity, dtype=float), 0.0, 100.0),
        limit_inputs=False, round_output=False,
    )
    return np.asarray(result.utci, dtype=float)


def add_utci(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` with a ``utci`` column, computed only where it is missing."""
    if "utci" in df and df["utci"].notna().all():
        return df
    utci = compute_utci(df["temperature"], df["humidity"], df["wind_speed"])
    if "utci" in df:
        utci = df["utci"].fillna(pd.Series(utci, index=df.index)).to_numpy()
    return df.assign(utci=utci)


def _validate(df: pd.DataFrame, source: str) -> None:
    missing = [c for c in SENSOR_COLUMNS if c not in df]
    if missing:
        raise ValueError(f"{source}: missing sensor columns {missing}")
    values = df[SENSOR_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = df.index[~np.isfinite(values).all(axis=1)]
        raise ValueError(f"{source}: {len(bad)} rows with missing or non-finite sensor values, "
                         f"first at {bad[0]}")


def _segments(index: pd.DatetimeIndex, step: pd.Timedelta, clip_id: pd.Series | None) -> np.ndarray:
    """Number the unbroken stretches: a new one at every clip and every time jump."""
    breaks = np.r_[True, np.diff(index.asi8) != step.value]
    if clip_id is not None:
        ids = clip_id.to_numpy()
        breaks |= np.r_[True, ids[1:] != ids[:-1]]
    return np.cumsum(breaks) - 1


def load_clips(path: str | Path = DEPLOYMENT / "evaluation_dataset" / "clips.csv",
               clip_ids: list[str] | None = None) -> pd.DataFrame:
    """Recorded clips as a DataFrame indexed by timestamp.

    Keeps the sensor, UTCI and label columns, checks the sensor values, sorts
    each clip by time and adds ``segment``: one number per unbroken stretch
    (normally one per clip). ``clip_ids`` keeps only those clips.
    """
    path = Path(path)
    df = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"],
                     dtype={"clip_id": str}, keep_default_na=False, na_values={"utci": [""]})
    if clip_ids is not None:
        df = df[df["clip_id"].isin(clip_ids)]
        unknown = set(clip_ids) - set(df["clip_id"])
        if unknown:
            raise KeyError(f"clips not in {path.name}: {sorted(unknown)}")
    keep = [c for c in LABEL_COLUMNS + SENSOR_COLUMNS + OPTIONAL_COLUMNS if c in df]
    df = df[keep]
    _validate(df, path.name)
    if "clip_id" in df:
        df = (df.rename_axis("timestamp").reset_index()
                .sort_values(["clip_id", "timestamp"], kind="stable").set_index("timestamp"))
    else:
        df = df.sort_index()
    df = add_utci(df)
    df["segment"] = _segments(df.index, STEP, df["clip_id"] if "clip_id" in df else None)
    return df


def load_manifest(path: str | Path = DEPLOYMENT / "evaluation_dataset" / "manifest.csv") -> pd.DataFrame:
    """One row per clip (category, openings, closings, ...), indexed by ``clip_id``."""
    return pd.read_csv(path, index_col="clip_id", dtype={"clip_id": str},
                       parse_dates=["start", "end"])


def iter_clips(df: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    """(clip_id, rows) for every clip in ``df``, in order of appearance."""
    for clip_id in df["clip_id"].unique():
        yield clip_id, df[df["clip_id"] == clip_id]


def readings(df: pd.DataFrame) -> Iterator[Reading]:
    """Rows of ``df`` as ``Reading``s, the form the device produces."""
    utci = df["utci"].to_numpy() if "utci" in df else np.full(len(df), np.nan)
    cols = [df[c].to_numpy(dtype=float) for c in SENSOR_COLUMNS]
    for t, temp, hum, wind, rain, u in zip(df.index, *cols, utci):
        yield Reading(t, temp, hum, wind, rain, None if np.isnan(u) else float(u))


def save_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write ``frame`` to ``path`` all at once: a crash or Ctrl+C never leaves half a file.

    It is written to a temporary file next to ``path`` first and then renamed
    over it; until the rename, any existing file at ``path`` stays as it was.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_csv(tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
