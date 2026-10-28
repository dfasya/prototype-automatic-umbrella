"""Load KMA AWS one-minute CSVs and feed them to the PCHIP upsampler.

The raw files are cp949-encoded with Korean headers, contain gaps and missing
values, and report wind direction as an angle. ``upsample_weather`` needs
contiguous, finite 60-second records and cannot interpolate angles, so this
module renames the columns, splits each variable into contiguous runs, and
interpolates wind direction through the sine and cosine of the angle.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from modules.simulation.pchip import Bounds, Mode, upsample_weather

__all__ = [
    "COLUMNS", "DEFAULT_MODES", "DEFAULT_BOUNDS",
    "load_aws", "add_wind_components", "contiguous_runs", "simulate", "load_simulated",
]

COLUMNS = {
    "지점": "station",
    "일시": "timestamp",
    "기온(°C)": "temperature",
    "1분 강수량(mm)": "rain",
    "강수유무(유무)": "rain_flag",
    "풍향(deg)": "wind_dir",
    "풍속(m/s)": "wind_speed",
    "현지기압(hPa)": "pressure",
    "해면기압(hPa)": "sea_pressure",
    "습도(%)": "humidity",
    "일사(MJ/m^2)": "solar",
    "일조(Sec)": "sunshine",
}

# Assumptions about how the AWS reports each variable; adjust if needed.
DEFAULT_MODES: dict[str, Mode] = {
    "temperature": "point",
    "pressure": "point",
    "sea_pressure": "point",
    "humidity": "point",
    "wind_speed": "mean",
    "dir_sin": "mean",  # direction as sine/cosine; replaced by wind_dir/wind_u/wind_v
    "dir_cos": "mean",
    "rain": "total",
}

DEFAULT_BOUNDS: dict[str, Bounds] = {
    "humidity": (0.0, 100.0),
    "wind_speed": (0.0, None),
    "dir_sin": (-1.0, 1.0),
    "dir_cos": (-1.0, 1.0),
}


def load_aws(
    source: str | Path | Iterable[str | Path] = "dataset",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read AWS CSVs (a directory, one file, or a list), deduplicated and sorted.

    Overlapping files (e.g. a yearly export and monthly exports) are merged;
    the solar and sunshine columns are dropped because they only hold
    sentinel values in this dataset.
    """
    if isinstance(source, (str, Path)):
        source = Path(source)
        paths = sorted(source.glob("*.csv")) if source.is_dir() else [source]
    else:
        paths = [Path(p) for p in source]
    if not paths:
        raise FileNotFoundError("No CSV files found.")

    frames = [pd.read_csv(p, encoding="cp949").rename(columns=COLUMNS) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = (df.drop(columns=["solar", "sunshine"], errors="ignore")
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp", ignore_index=True))
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] < pd.Timestamp(end)]
    return df.reset_index(drop=True)


def load_simulated(
    directory: str | Path = "simulated_dataset",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    raw_source: str | Path | None = "dataset",
) -> pd.DataFrame:
    """Read the monthly CSVs written by simulation.ipynb into one time-indexed frame.

    Only files whose month overlaps ``[start, end)`` are read. If the files lack
    ``rain_flag`` and ``raw_source`` is given, each row gets the flag of the
    source minute containing it (KMA labels minutes by their end). Adds a
    ``segment`` column numbering the contiguous runs: rows more than one step
    apart start a new segment, so sequence models never bridge a data gap.
    """
    directory = Path(directory)
    start = None if start is None else pd.Timestamp(start)
    end = None if end is None else pd.Timestamp(end)
    files = []
    for path in sorted(directory.glob("*.csv")):
        month = pd.Period(path.stem, freq="M")
        if start is not None and month.end_time < start:
            continue
        if end is not None and month.start_time >= end:
            continue
        files.append(path)
    if not files:
        raise FileNotFoundError(f"No monthly CSVs in {directory} for the requested range.")

    df = pd.concat(
        (pd.read_csv(p, index_col="timestamp", parse_dates=["timestamp"]) for p in files)
    ).sort_index()
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index < end]
    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps across the simulated files.")

    if "rain_flag" not in df and raw_source is not None:
        raw = load_aws(raw_source).set_index("timestamp")
        minute = df.index.floor("min") + pd.Timedelta(minutes=1)
        df["rain_flag"] = raw["rain_flag"].reindex(minute).fillna(0.0).to_numpy()

    values = df.select_dtypes("number").to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Simulated data contains NaN or infinite values.")

    step = df.index.to_series().diff().median()
    df["segment"] = (df.index.to_series().diff() != step).cumsum().to_numpy() - 1
    df.attrs = {"files": [p.name for p in files], "step": step}
    return df


def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``dir_sin``/``dir_cos``, the sine and cosine of the direction the wind blows from."""
    rad = np.deg2rad(df["wind_dir"])
    return df.assign(dir_sin=np.sin(rad), dir_cos=np.cos(rad))


def contiguous_runs(timestamps: pd.Series, valid: pd.Series) -> list[np.ndarray]:
    """Row positions of maximal runs that are valid and spaced exactly 60 s apart."""
    ts = pd.DatetimeIndex(timestamps)
    valid = np.asarray(valid, dtype=bool)
    step_ok = np.r_[False, np.diff(ts.as_unit("ns").asi8) == 60 * 10**9]
    breaks = ~valid | ~np.r_[False, valid[:-1]] | ~step_ok
    run_id = np.cumsum(breaks)
    positions = np.flatnonzero(valid)
    return np.split(positions, np.flatnonzero(np.diff(run_id[positions])) + 1)


def _upsample_column(
    df: pd.DataFrame, column: str, mode: Mode, bounds: Bounds | None,
    step_seconds: int, interval_label: str, min_points: int,
) -> tuple[pd.Series, int]:
    runs = contiguous_runs(df["timestamp"], df[column].notna())
    needed = max(min_points, 2 if mode == "point" else 1)
    pieces, skipped = [], 0
    for run in runs:
        if len(run) < needed:
            skipped += len(run)
            continue
        out = upsample_weather(
            df.iloc[run], {column: mode},
            step_seconds=step_seconds, interval_label=interval_label,
            bounds=None if bounds is None else {column: bounds},
        )
        pieces.append(out[column])
    if not pieces:
        return pd.Series(dtype=float, name=column, index=pd.DatetimeIndex([])), skipped
    series = pd.concat(pieces)
    # Point runs that end exactly where the next begins share a timestamp.
    return series[~series.index.duplicated(keep="first")], skipped


def simulate(
    df: pd.DataFrame,
    modes: Mapping[str, Mode] | None = None,
    bounds: Mapping[str, Bounds] | None = None,
    step_seconds: int = 5,
    interval_label: str = "end",
    min_points: int = 2,
    fill_zero: Iterable[str] = ("rain", "rain_flag"),
    dropna: bool = True,
    hold: Iterable[str] = ("rain_flag",),
) -> pd.DataFrame:
    """Upsample every selected column to ``step_seconds`` resolution.

    Each column is processed independently over its own contiguous, non-missing
    runs, so a gap in one variable does not discard the others. Runs shorter
    than ``min_points`` are skipped. KMA labels minute data by interval end,
    hence the ``interval_label`` default. Wind direction is rebuilt from the
    interpolated ``dir_sin``/``dir_cos`` when both are present; with
    ``wind_speed`` it also gives ``wind_u``/``wind_v``, the eastward and
    northward wind in m/s.

    ``hold`` columns (e.g. the 0/10 precipitation detector) are categorical, so
    each sub-minute row repeats the value of the source minute containing it.

    Missing values in ``fill_zero`` columns become 0.0 (no rain recorded). With
    ``dropna``, rows still missing any other value are then removed; these come
    from source gaps and from the first/last minute, where point and interval
    columns do not overlap.
    """
    modes = dict(DEFAULT_MODES if modes is None else modes)
    bounds = DEFAULT_BOUNDS if bounds is None else bounds
    if {"dir_sin", "dir_cos"} & set(modes) and "dir_sin" not in df:
        df = add_wind_components(df)

    columns, skipped = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for column, mode in modes.items():
            col_bounds = bounds.get(column) if mode != "total" else None
            series, skipped[column] = _upsample_column(
                df, column, mode, col_bounds, step_seconds, interval_label, min_points)
            columns.append(series)

    result = pd.concat(columns, axis=1).sort_index()
    result.index.name = "timestamp"
    if {"dir_sin", "dir_cos"} <= set(result):
        sin, cos = result.pop("dir_sin"), result.pop("dir_cos")
        direction = np.rad2deg(np.arctan2(sin, cos)) % 360.0
        result["wind_dir"] = direction.where(np.hypot(sin, cos) > 0)
        if "wind_speed" in result:
            # Meteorological convention: wind_dir is where the wind blows from.
            rad = np.deg2rad(result["wind_dir"])
            result["wind_u"] = -result["wind_speed"] * np.sin(rad)
            result["wind_v"] = -result["wind_speed"] * np.cos(rad)
    # Row t lies in the source minute starting at floor(t); KMA labels it by its end.
    minute_label = result.index.floor("min")
    if interval_label == "end":
        minute_label += pd.Timedelta(minutes=1)
    source_by_time = df.set_index("timestamp")
    for column in hold:
        if column in source_by_time:
            result[column] = source_by_time[column].reindex(minute_label).to_numpy()
    nan_counts = result.isna().sum().to_dict()
    filled = [c for c in fill_zero if c in result]
    result[filled] = result[filled].fillna(0.0)
    rows_before = len(result)
    if dropna:
        result = result.dropna()
    result.attrs = {
        "product": "deterministically interpolated/disaggregated data; not measured truth",
        "source_seconds": 60, "step_seconds": step_seconds,
        "input_interval_label": interval_label, "output_interval_label": "start",
        "modes": modes, "bounds": {k: list(v) for k, v in bounds.items()},
        "input_rows": len(df), "output_rows": len(result),
        "skipped_source_rows": skipped,
        "nan_before_cleaning": nan_counts, "zero_filled_columns": filled,
        "dropped_rows": rows_before - len(result),
    }
    return result
