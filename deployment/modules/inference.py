"""Real-time umbrella inference: one 20 s sensor reading in, one decision out.

    reading -> UTCI -> window features (last 15 min) -> HMM forward step
            -> P(OPEN in 5 min) -> threshold + hysteresis + hold -> OPEN / CLOSED

``Umbrella`` keeps all state between readings (feature buffers, filtered state
probabilities, current position), so it runs on the device unchanged. A time
jump (anything other than one 20 s step) is treated as a restart, like the
training data treats a data gap. Everything is causal: no reading after the
current one is ever used.

The model is a plain ``.npz`` of arrays (``export_model`` converts the
``models/umbrella_hmm.joblib`` bundle from training.ipynb), so inference needs
only numpy and pandas.

Command line (from ``deployment/``) replays recorded clips through it:

    python -m modules.inference                        # all clips in evaluation_dataset/
    python -m modules.inference --clip 20240709_1130   # one clip, decision per row
    python -m modules.inference --export ../models/umbrella_hmm.joblib
"""
from __future__ import annotations

import argparse
import math
import sys
import types
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # run as a file: python modules/inference.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from modules.dataset import DEPLOYMENT, STEP, Reading, load_clips, load_manifest, readings, save_csv
else:
    from .dataset import DEPLOYMENT, STEP, Reading, load_clips, load_manifest, readings, save_csv

__all__ = ["MODEL_PATH", "UmbrellaModel", "WindowFeatures", "Decision", "Umbrella",
           "run_stream", "summarize", "export_model"]

MODEL_PATH = DEPLOYMENT / "models" / "umbrella_hmm.npz"
_LOG_2PI = math.log(2 * math.pi)


# ------------------------------------------------------------------ model --

class UmbrellaModel:
    """The trained HMM, its scaler, per-state OPEN rates and decision settings."""

    def __init__(self, arrays: dict):
        self.features = [str(f) for f in arrays["features"]]
        self.startprob = np.asarray(arrays["startprob"], dtype=float)
        self.transmat = np.asarray(arrays["transmat"], dtype=float)
        self.means = np.asarray(arrays["means"], dtype=float)
        self.covars = np.asarray(arrays["covars"], dtype=float)
        self.scaler_mean = np.asarray(arrays["scaler_mean"], dtype=float)
        self.scaler_std = np.asarray(arrays["scaler_std"], dtype=float)
        self.state_open_rates = np.asarray(arrays["state_open_rates"], dtype=float)
        self.threshold = float(arrays["threshold"])
        self.close_margin = float(arrays["close_margin"])
        self.min_hold_minutes = float(arrays["min_hold_minutes"])
        self.horizon_minutes = float(arrays["horizon_minutes"])
        self.step = pd.Timedelta(seconds=float(arrays["step_seconds"]))
        self.trained_from = str(arrays["trained_from"])
        self.trained_to = str(arrays["trained_to"])

        with np.errstate(divide="ignore"):
            self.log_start = np.log(self.startprob)
            self.log_trans = np.log(self.transmat)
        chol = np.linalg.cholesky(self.covars)
        self._chol_inv = np.linalg.inv(chol)
        self._log_norm = -0.5 * (len(self.features) * _LOG_2PI
                                 + 2 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1))

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "UmbrellaModel":
        path = Path(path)
        if path.suffix == ".joblib":
            return cls(_read_joblib(path))
        with np.load(path, allow_pickle=False) as npz:
            return cls({k: npz[k] for k in npz.files})

    @property
    def n_states(self) -> int:
        return len(self.startprob)

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """log N(x | state k) for every state; ``x`` is one row of raw features."""
        z = (np.asarray(x, dtype=float) - self.scaler_mean) / self.scaler_std
        w = np.einsum("kij,kj->ki", self._chol_inv, z - self.means)
        return self._log_norm - 0.5 * (w * w).sum(axis=1)

    def describe(self) -> str:
        return (f"{self.n_states}-state HMM, trained {self.trained_from} to {self.trained_to}; "
                f"P(OPEN in {self.horizon_minutes:g} min), open at >= {self.threshold:g}, "
                f"close below {self.threshold - self.close_margin:g}, "
                f"hold {self.min_hold_minutes:g} min")


def _read_joblib(path: Path) -> dict:
    """Arrays from a training bundle, without needing the training code.

    The bundle pickles ``modules.model.hmm.RidgeGaussianHMM``; that name is
    mapped to a stand-in that only keeps the fitted attributes.
    """
    import joblib

    class _HMM:
        def __setstate__(self, state):
            self.__dict__.update(state)

    names = ["modules.model", "modules.model.hmm"]
    saved = {n: sys.modules.get(n) for n in names}
    stub = types.ModuleType("modules.model.hmm")
    stub.RidgeGaussianHMM = stub.GaussianHMM = _HMM
    sys.modules["modules.model"] = types.ModuleType("modules.model")
    sys.modules["modules.model.hmm"] = stub
    try:
        bundle = joblib.load(path)
    finally:
        for n, module in saved.items():
            if module is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = module
    hmm = bundle["model"]
    if hmm.covariance_type != "full":
        raise ValueError(f"expected a full-covariance HMM, got {hmm.covariance_type!r}")
    return {
        "features": np.array(bundle["features"]),
        "startprob": hmm.startprob_, "transmat": hmm.transmat_,
        "means": hmm.means_, "covars": hmm._covars_,
        "scaler_mean": bundle["scaler_mean"], "scaler_std": bundle["scaler_std"],
        "state_open_rates": bundle["state_open_rates"],
        "threshold": bundle["threshold"],
        "close_margin": bundle.get("close_margin", 0.0),
        "min_hold_minutes": bundle.get("min_hold_minutes", 0.0),
        "horizon_minutes": bundle["horizon_minutes"],
        "step_seconds": pd.Timedelta(bundle["step"]).total_seconds(),
        "trained_from": str(pd.Timestamp(bundle["trained_from"]).date()),
        "trained_to": str(pd.Timestamp(bundle["trained_to"]).date()),
    }


def export_model(source: str | Path, target: str | Path = MODEL_PATH) -> Path:
    """Convert a ``.joblib`` training bundle into the ``.npz`` the device loads.

    Written to a temporary file and renamed over ``target`` at the end, so an
    interrupted export leaves the existing model intact.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = _read_joblib(Path(source))
    tmp = target.with_name(f".{target.stem}.tmp.npz")  # np.savez needs the .npz suffix
    try:
        np.savez(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


# --------------------------------------------------------------- features --

class WindowFeatures:
    """Trailing-window features, updated one reading at a time.

    Same definitions as ``add_window_features`` in training: 10-min mean/max
    UTCI, 5/10-min share of wet readings, rain recency (1 while wet, then
    exp(-minutes/5), 0 after 30 min dry), humidity and its 15-min change,
    5-min mean/max wind, and the time of day on a circle.
    """

    WET_CAP_MINUTES = 30.0
    RAIN_DECAY_MINUTES = 5.0

    def __init__(self, step: pd.Timedelta = STEP):
        self.step = step
        self._lag = int(pd.Timedelta(minutes=15) / step)
        self.reset()

    def reset(self) -> None:
        self._recent = deque()                       # (t, utci, wet, wind) of the last 10 min
        self._humidity = deque(maxlen=self._lag + 1)
        self._first_humidity = None
        self._last_wet = None
        self.since = None                            # when the current stretch started

    def update(self, r: Reading) -> dict[str, float]:
        t = r.timestamp
        if self.since is None:
            self.since = t
        wet = 1.0 if r.rain_flag > 0 else 0.0
        self._recent.append((t, r.utci, wet, r.wind_speed))
        while self._recent[0][0] <= t - pd.Timedelta(minutes=10):
            self._recent.popleft()
        last5 = [row for row in self._recent if row[0] > t - pd.Timedelta(minutes=5)]

        if wet:
            self._last_wet = t
        if self._last_wet is None:
            since_wet = self.WET_CAP_MINUTES
        else:
            since_wet = min((t - self._last_wet).total_seconds() / 60, self.WET_CAP_MINUTES)

        if self._first_humidity is None:
            self._first_humidity = r.humidity
        self._humidity.append(r.humidity)
        earlier = self._humidity[0] if len(self._humidity) > self._lag else self._first_humidity

        hours = t.hour + t.minute / 60 + t.second / 3600
        angle = 2 * math.pi * hours / 24
        utci = [row[1] for row in self._recent]
        return {
            "utci_mean_10m": sum(utci) / len(utci),
            "utci_max_10m": max(utci),
            "wet_share_5m": sum(row[2] for row in last5) / len(last5),
            "wet_share_10m": sum(row[2] for row in self._recent) / len(self._recent),
            "rain_recency": (math.exp(-since_wet / self.RAIN_DECAY_MINUTES)
                             if since_wet < self.WET_CAP_MINUTES else 0.0),
            "humidity": r.humidity,
            "humidity_chg_15m": r.humidity - earlier,
            "wind_mean_5m": sum(row[3] for row in last5) / len(last5),
            "wind_max_5m": max(row[3] for row in last5),
            "hour_sin": math.sin(angle),
            "hour_cos": math.cos(angle),
        }


# -------------------------------------------------------------- controller --

@dataclass(frozen=True)
class Decision:
    timestamp: pd.Timestamp
    p_open: float        # P(umbrella should be OPEN in horizon_minutes)
    open: bool           # what the umbrella does now
    state: int           # most likely hidden weather state
    restarted: bool      # first reading after a start or a time jump
    warming_up: bool     # less than 15 min of history: windows still filling


class Umbrella:
    """Stateful controller: feed it ``Reading``s every 20 s, get ``Decision``s."""

    def __init__(self, model: UmbrellaModel):
        self.model = model
        self.features = WindowFeatures(model.step)
        self._hold = int(round(pd.Timedelta(minutes=model.min_hold_minutes) / model.step))
        self.reset()

    def reset(self) -> None:
        """Forget everything, as after a power cycle or a data gap."""
        self.features.reset()
        self._log_alpha = None
        self._last_time = None
        self.is_open = False
        self._since_switch = 0
        self.last_features: dict[str, float] = {}    # inputs of the latest step, for inspection
        self.last_posterior = np.full(self.model.n_states, np.nan)

    def step(self, reading: Reading) -> Decision:
        reading = reading.with_utci()
        restarted = self._last_time is None or reading.timestamp - self._last_time != self.model.step
        if restarted:
            self.reset()
        self._last_time = reading.timestamp

        values = self.features.update(reading)
        x = np.array([values[f] for f in self.model.features])
        ll = self.model.log_likelihood(x)
        if self._log_alpha is None:
            alpha = self.model.log_start + ll
        else:
            with np.errstate(invalid="ignore"):
                joint = self._log_alpha[:, None] + self.model.log_trans
                peak = joint.max(axis=0)
                safe = np.where(np.isfinite(peak), peak, 0.0)
                alpha = np.where(np.isfinite(peak),
                                 safe + np.log(np.exp(joint - safe).sum(axis=0)), -np.inf) + ll
        peak = alpha.max()
        self._log_alpha = alpha - (peak + math.log(np.exp(alpha - peak).sum()))
        posterior = np.exp(self._log_alpha)
        p_open = float(posterior @ self.model.state_open_rates)
        self.last_features, self.last_posterior = values, posterior

        m = self.model
        if restarted:
            self.is_open = p_open >= m.threshold
            self._since_switch = self._hold
        elif self._since_switch >= self._hold:
            limit = m.threshold - m.close_margin if self.is_open else m.threshold
            if (p_open >= limit) != self.is_open:
                self.is_open = not self.is_open
                self._since_switch = 0
        self._since_switch += 1

        return Decision(reading.timestamp, p_open, self.is_open, int(posterior.argmax()), restarted,
                        reading.timestamp - self.features.since < pd.Timedelta(minutes=15))


def run_stream(model: UmbrellaModel, df: pd.DataFrame) -> pd.DataFrame:
    """Replay a recorded stream through ``Umbrella``, one reading at a time.

    ``df`` is indexed by timestamp with the sensor columns (``load_clips``).
    The controller restarts at every new ``segment`` (clip) and every time
    jump. Returns the decisions alongside the input's label columns.
    """
    umbrella = Umbrella(model)
    segment = df["segment"].to_numpy() if "segment" in df else np.zeros(len(df), dtype=int)
    rows = []
    for i, reading in enumerate(readings(df)):
        if i and segment[i] != segment[i - 1]:
            umbrella.reset()
        rows.append(umbrella.step(reading))
    out = pd.DataFrame([vars(d) for d in rows]).set_index("timestamp")
    out["decision"] = np.where(out["open"], "OPEN", "CLOSED")
    extra = [c for c in ("clip_id", "phase", "label", "reason", "label_ahead") if c in df]
    return pd.concat([df[extra].reset_index(drop=True).set_axis(out.index), out], axis=1)


# ----------------------------------------------------------------- report --

def _scores(truth: np.ndarray, pred: np.ndarray) -> dict:
    tp = int((truth & pred).sum())
    precision = tp / max(int(pred.sum()), 1)
    recall = tp / max(int(truth.sum()), 1)
    return {"rows": len(truth), "open_share": truth.mean() if len(truth) else np.nan,
            "accuracy": (truth == pred).mean() if len(truth) else np.nan,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / max(int(pred.sum() + truth.sum()), 1),
            "false_open": (pred & ~truth).sum() / max(int((~truth).sum()), 1)}


def summarize(predictions: pd.DataFrame, manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    """Scores on the ``eval`` rows against ``label_ahead``, against persistence.

    Persistence ("the label stays as it is now") is the baseline the model has
    to beat. With a ``manifest`` the scores are also split by clip category.
    """
    scored = predictions[(predictions["phase"] == "eval") & (predictions["label_ahead"] != "")]
    truth = (scored["label_ahead"] == "OPEN").to_numpy()
    groups = {"all clips": np.ones(len(scored), dtype=bool)}
    if manifest is not None:
        category = manifest["category"].reindex(scored["clip_id"]).to_numpy()
        groups.update({c: category == c for c in pd.unique(category)})
    table = {}
    for name, mask in groups.items():
        table[(name, "model")] = _scores(truth[mask], scored["open"].to_numpy()[mask])
        table[(name, "persistence")] = _scores(truth[mask], (scored["label"] == "OPEN").to_numpy()[mask])
    return pd.DataFrame(table).T


def _moves_per_day(predictions: pd.DataFrame, column: str) -> float:
    moves, hours = 0, 0.0
    for _, part in predictions.groupby("clip_id"):
        state = part[column].to_numpy()
        moves += int((state[1:] != state[:-1]).sum())
        hours += len(part) * STEP / pd.Timedelta(hours=1)
    return moves / hours * 24


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recorded clips through the umbrella controller.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH,
                        help=".npz (or .joblib) model (default: models/umbrella_hmm.npz)")
    parser.add_argument("--clips", type=Path, default=DEPLOYMENT / "evaluation_dataset" / "clips.csv",
                        help="clips to replay (default: evaluation_dataset/clips.csv)")
    parser.add_argument("--clip", action="append", default=None,
                        help="only this clip_id, repeatable (default: all clips)")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the per-row decisions to this CSV (default: not saved)")
    parser.add_argument("--export", type=Path, metavar="JOBLIB", default=None,
                        help="convert a training .joblib bundle to --model (.npz) and exit (default: off)")
    args = parser.parse_args()

    if args.export:
        print("wrote", export_model(args.export, args.model))
        return

    model = UmbrellaModel.load(args.model)
    print(model.describe())
    clips = load_clips(args.clips, args.clip)
    predictions = run_stream(model, clips)
    if args.out:
        print("wrote", save_csv(predictions, args.out))

    manifest_path = args.clips.with_name("manifest.csv")
    manifest = load_manifest(manifest_path) if manifest_path.exists() else None
    n_clips = predictions["clip_id"].nunique()
    print(f"\n{n_clips} clips, {len(predictions):,} readings")
    if n_clips == 1:
        changes = predictions[predictions["decision"] != predictions["decision"].shift()]
        print("\nmoves (model):")
        print(changes[["phase", "p_open", "decision", "label"]].round(3).to_string())
    pd.set_option("display.width", 120)
    print("\nscored rows (phase == eval) vs the label", f"{model.horizon_minutes:g} min ahead:")
    print(summarize(predictions, manifest)[["rows", "open_share", "precision", "recall", "f1", "false_open"]]
          .round(3).to_string())
    print("(false_open: share of should-be-CLOSED rows where it was open)")
    print(f"\nmoves per 24 h: model {_moves_per_day(predictions, 'open'):.1f}, "
          f"label {_moves_per_day(predictions, 'label'):.1f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped; nothing was overwritten")
        sys.exit(130)
