"""Gaussian HMM for the umbrella, evaluated the way it would run in the field.

The HMM is unsupervised: it learns weather states from sensor features only.
Each state is then given an OPEN rate from the training labels, and at run
time ``P(OPEN) = sum_k P(state k | observations so far) * rate_k``.

Everything here is causal. State probabilities come from forward filtering
(no look-ahead), and ``walk_forward`` retrains on past training rows only
before scoring the next block of evaluation hours.
"""
from __future__ import annotations

import copy
import logging
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from numba import njit
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from modules.dataset.split import sequence_lengths

__all__ = [
    "FEATURES", "add_time_features", "future_target", "fit_hmm", "filter_states",
    "state_open_rates", "best_threshold", "walk_forward", "apply_hysteresis", "evaluate", "evaluate_actions", "action_episodes",
    "train_final", "predict_open",
]

log = logging.getLogger(__name__)

# Current readings only, from the device's sensors (no barometer, no rain gauge).
FEATURES = ["temperature", "humidity", "wind_u", "wind_v", "rain_flag",
            "hour_sin", "hour_cos"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time of day as a point on a circle, so 23:59 sits next to 00:00."""
    hours = (df.index.hour + df.index.minute / 60 + df.index.second / 3600).to_numpy()
    angle = 2 * np.pi * hours / 24
    return df.assign(hour_sin=np.sin(angle), hour_cos=np.cos(angle))


def future_target(label: pd.Series, horizon: pd.Timedelta) -> pd.Series:
    """Whether the label ``horizon`` from now is OPEN; NaN where that row is missing."""
    is_open = (label == "OPEN").astype(float)
    future = is_open.reindex(label.index + horizon).to_numpy()
    return pd.Series(future, index=label.index, name="target")


# ---------------------------------------------------------------- scaling --

class _Scaler:
    def __init__(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def _rescaled(model: GaussianHMM, old: _Scaler, new: _Scaler) -> GaussianHMM:
    """Copy of a model fitted in ``old`` scaled units, moved into ``new`` scaled units."""
    model = copy.deepcopy(model)
    ratio = old.std / new.std
    model.means_ = (model.means_ * old.std + old.mean - new.mean) / new.std
    model.covars_ = model.covars_ * np.outer(ratio, ratio)[None, :, :]
    return model


def _check_healthy(model: GaussianHMM) -> None:
    """Raise if EM left unused states, non-finite values or invalid covariances."""
    if not (np.isfinite(model.means_).all() and np.isfinite(model.transmat_).all()
            and np.isfinite(model.startprob_).all()):
        raise ArithmeticError("non-finite parameters")
    if not np.allclose(model.transmat_.sum(axis=1), 1.0):
        raise ArithmeticError("a state is never used")
    if np.any(np.linalg.eigvalsh(model.covars_) <= 0):
        raise ArithmeticError("covariance is not positive definite")


# ---------------------------------------------------------------- fitting --

class RidgeGaussianHMM(GaussianHMM):
    """GaussianHMM that adds ``min_covar`` to every covariance diagonal after each M-step.

    hmmlearn applies ``min_covar`` only at initialisation. Features that are
    exactly constant within a state (e.g. ``wet_share`` = 0 in dry weather)
    then collapse that state's covariance to a singular matrix and EM breaks.
    The ridge keeps every covariance positive definite.
    """

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        self._covars_ = self._covars_ + self.min_covar * np.eye(self.n_features)[None, :, :]


def _new_model(n_states: int, n_iter: int, seed: int, init_params: str) -> GaussianHMM:
    return RidgeGaussianHMM(n_components=n_states, covariance_type="full", n_iter=n_iter,
                            tol=1e-3, random_state=seed, init_params=init_params, min_covar=1e-2)


def fit_hmm(
    X: np.ndarray, lengths: np.ndarray, n_states: int = 5, n_iter: int = 50,
    n_restarts: int = 3, seed: int = 0, init: GaussianHMM | None = None,
) -> GaussianHMM:
    """Fit a full-covariance GaussianHMM.

    With ``init`` the fit warm-starts from that model (one run of ``n_iter``);
    otherwise it keeps the best of ``n_restarts`` random initialisations.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if init is not None:
            model = _new_model(n_states, n_iter, seed, init_params="")
            model.startprob_, model.transmat_ = init.startprob_, init.transmat_
            model.means_, model.covars_ = init.means_, init.covars_
            model.fit(X, lengths)
            _check_healthy(model)
            return model
        best, best_score = None, -np.inf
        for r in range(n_restarts):
            try:
                model = _new_model(n_states, n_iter, seed + r, init_params="stmc").fit(X, lengths)
                _check_healthy(model)
            except (ValueError, ArithmeticError, np.linalg.LinAlgError):
                continue
            score = model.monitor_.history[-1]
            if np.isfinite(score) and score > best_score:
                best, best_score = model, score
        if best is None:
            raise ArithmeticError("Every HMM restart failed.")
        return best


# -------------------------------------------------------------- filtering --

@njit(cache=True)
def _forward_filter(log_start, log_trans, framelogprob, reset):
    n, k = framelogprob.shape
    out = np.empty((n, k))
    prev = np.empty(k)
    cur = np.empty(k)
    for t in range(n):
        for j in range(k):
            if t == 0 or reset[t]:
                cur[j] = log_start[j]
            else:
                m = -np.inf
                for i in range(k):
                    m = max(m, prev[i] + log_trans[i, j])
                s = 0.0
                if np.isfinite(m):
                    for i in range(k):
                        s += np.exp(prev[i] + log_trans[i, j] - m)
                    cur[j] = m + np.log(s)
                else:
                    cur[j] = -np.inf
            cur[j] += framelogprob[t, j]
        m = cur.max()
        s = 0.0
        for j in range(k):
            s += np.exp(cur[j] - m)
        norm = m + np.log(s)
        for j in range(k):
            prev[j] = cur[j] - norm
            out[t, j] = np.exp(prev[j])
    return out


def filter_states(model: GaussianHMM, X: np.ndarray, reset: np.ndarray) -> np.ndarray:
    """P(state | observations up to now) for every row; ``reset`` restarts a sequence."""
    with np.errstate(divide="ignore"):
        log_start = np.log(model.startprob_)
        log_trans = np.log(model.transmat_)
    framelogprob = model._compute_log_likelihood(X)
    return _forward_filter(log_start, log_trans, framelogprob, np.asarray(reset, dtype=np.bool_))


def _resets(index: pd.DatetimeIndex, step: pd.Timedelta) -> np.ndarray:
    lengths = sequence_lengths(index, step)
    reset = np.zeros(len(index), dtype=bool)
    reset[np.r_[0, np.cumsum(lengths)[:-1]]] = True
    return reset


def state_open_rates(posteriors: np.ndarray, target: np.ndarray, prior: float = 1.0) -> np.ndarray:
    """Posterior-weighted OPEN rate of each state, with a Beta(prior, prior) prior."""
    valid = np.isfinite(target)
    weights = posteriors[valid]
    opened = weights.T @ target[valid]
    return (opened + prior) / (weights.sum(axis=0) + 2 * prior)


def best_threshold(target: np.ndarray, p_open: np.ndarray) -> float:
    """Threshold in 0.05-0.95 that maximises F1 of OPEN on these rows."""
    valid = np.isfinite(target)
    y, p = target[valid] == 1, p_open[valid]
    candidates = np.round(np.arange(0.05, 0.951, 0.05), 2)
    f1 = []
    for t in candidates:
        pred = p >= t
        tp = np.sum(pred & y)
        f1.append(2 * tp / max(pred.sum() + y.sum(), 1))
    return float(candidates[int(np.argmax(f1))])


# ----------------------------------------------------------- walk forward --

def walk_forward(
    data: pd.DataFrame,
    features: list[str] = FEATURES,
    n_states: int = 5,
    horizon_minutes: float = 5.0,
    initial_days: int = 14,
    retrain_days: int = 7,
    window_days: int | None = 365,
    n_iter: int = 50,
    warm_iter: int = 10,
    n_restarts: int = 3,
    seed: int = 0,
    tune_threshold: bool = False,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate deployment: retrain every ``retrain_days`` and score the days after.

    ``data`` needs the feature columns, ``label`` (OPEN/CLOSED), ``reason`` and
    ``split`` from ``hourly_split``. For each fold the model is fitted on the
    "train" rows of the last ``window_days`` days that have data (``None`` =
    all past days), warm-started from the previous fold. It then filters the
    whole stream of the next ``retrain_days`` days, as the device would see it,
    and scores the "eval" rows against the label ``horizon_minutes`` ahead.
    With ``tune_threshold`` each fold's OPEN threshold maximises F1 on its own
    training rows; otherwise it is 0.5.

    Returns per-row predictions for the evaluation rows and a per-fold log.
    """
    step = data.index.to_series().diff().median()
    horizon = pd.Timedelta(minutes=horizon_minutes)
    data = data.assign(target=future_target(data["label"], horizon),
                       target_reason=data["reason"].reindex(data.index + horizon).to_numpy(),
                       now=(data["label"] == "OPEN").astype(float))
    days = data.index.normalize().unique()
    day_of_row = data.index.normalize()
    is_train = (data["split"] == "train").to_numpy()
    train_target_ok = data["split"].reindex(data.index + horizon).eq("train").to_numpy()

    folds = range(initial_days, len(days), retrain_days)
    if progress:
        from tqdm.auto import tqdm
        folds = tqdm(folds, desc="walk-forward", unit="fold")

    model, scaler, predictions, fold_log = None, None, [], []
    for fold, start in enumerate(folds):
        train_days = days[max(0, start - window_days) if window_days else 0:start]
        eval_days = days[start:start + retrain_days]
        train_mask = is_train & day_of_row.isin(train_days)
        train = data[train_mask]
        X_raw = train[features].to_numpy()
        new_scaler = _Scaler(X_raw)
        X = new_scaler(X_raw)
        lengths = sequence_lengths(train.index, step)

        # Warm start from last week's model; if that breaks, refit from scratch;
        # if that breaks too, keep serving last week's model (with its own scaler).
        status = "warm start" if model is not None else "fresh fit"
        fitted = None
        if model is not None:
            try:
                fitted = fit_hmm(X, lengths, n_states, warm_iter, seed=seed,
                                 init=_rescaled(model, scaler, new_scaler))
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                status = f"refit from scratch ({exc})"
        if fitted is None:
            try:
                fitted = fit_hmm(X, lengths, n_states, n_iter, n_restarts, seed + fold)
            except ArithmeticError as exc:
                if model is None:
                    raise
                status = f"kept previous model ({exc})"
        if fitted is not None:
            model, scaler = fitted, new_scaler
        else:
            X = scaler(X_raw)
        if status != "warm start":
            log.warning("fold %d: %s", fold, status)

        # OPEN rate per state, from training rows whose target is also a training row.
        train_post = filter_states(model, scaler(X_raw), _resets(train.index, step))
        target = np.where(train_target_ok[train_mask], train["target"].to_numpy(), np.nan)
        rates = state_open_rates(train_post, target)
        threshold = best_threshold(target, train_post @ rates) if tune_threshold else 0.5

        # Run the filter over the whole stream of the next days and score the eval rows.
        stream = data[day_of_row.isin(eval_days)]
        post = filter_states(model, scaler(stream[features].to_numpy()), _resets(stream.index, step))
        scored = (stream["split"] == "eval").to_numpy() & stream["target"].notna().to_numpy()
        p_open = post @ rates
        predictions.append(pd.DataFrame({
            "fold": fold,
            "p_open": p_open[scored],
            "threshold": threshold,
            "state": post[scored].argmax(axis=1),
            "target": stream["target"].to_numpy()[scored],
            "now": stream["now"].to_numpy()[scored],
            "reason": stream["target_reason"].to_numpy()[scored],
        }, index=stream.index[scored]))
        fold_log.append({
            "fold": fold, "train_from": train_days[0], "eval_from": eval_days[0],
            "eval_to": eval_days[-1], "train_rows": len(train), "eval_rows": int(scored.sum()),
            "status": status, "log_likelihood_per_row": model.score(X, lengths) / len(X),
            "state_open_rates": np.round(rates, 3).tolist(), "threshold": threshold,
        })

    return pd.concat(predictions), pd.DataFrame(fold_log).set_index("fold")


def apply_hysteresis(
    predictions: pd.DataFrame, margin: float = 0.2, min_hold_minutes: float = 0.0,
) -> np.ndarray:
    """OPEN/CLOSED decisions that open at ``threshold`` but close only below ``threshold - margin``.

    ``predictions`` needs ``p_open`` and ``threshold``. Each unbroken stretch
    starts from ``p_open >= threshold``. After every switch the umbrella holds
    its position for at least ``min_hold_minutes``. Both stop it flapping when
    P(OPEN) hovers around the threshold.
    """
    step = predictions.index.to_series().diff().median()
    p = predictions["p_open"].to_numpy()
    t = np.broadcast_to(predictions["threshold"].to_numpy(dtype=float), p.shape)
    start = np.zeros(len(p), dtype=bool)
    start[np.r_[0, np.cumsum(sequence_lengths(predictions.index, step))[:-1]]] = True
    hold = int(round(pd.Timedelta(minutes=min_hold_minutes) / step))
    return _hysteresis(p, t, start, margin, hold)


@njit(cache=True)
def _hysteresis(p, t, start, margin, hold):
    out = np.empty(len(p), dtype=np.bool_)
    state = False
    since_switch = 0
    for i in range(len(p)):
        if start[i]:
            state = p[i] >= t[i]
            since_switch = hold
        elif since_switch >= hold:
            new = p[i] >= (t[i] - margin if state else t[i])
            if new != state:
                state = new
                since_switch = 0
        since_switch += 1
        out[i] = state
    return out


# ------------------------------------------------------- deployed model --

def train_final(
    data: pd.DataFrame,
    features: list[str] = FEATURES,
    n_states: int = 5,
    horizon_minutes: float = 5.0,
    window_days: int | None = 365,
    n_iter: int = 50,
    n_restarts: int = 3,
    seed: int = 0,
    tune_threshold: bool = False,
) -> dict:
    """Fit the model to deploy on the "train" rows of the latest ``window_days`` days.

    Returns a bundle with everything ``predict_open`` needs: the HMM, the
    scaler statistics, the per-state OPEN rates and the settings used.
    """
    step = data.index.to_series().diff().median()
    horizon = pd.Timedelta(minutes=horizon_minutes)
    days = data.index.normalize().unique()
    keep_days = days[-window_days:] if window_days else days
    train = data[(data["split"] == "train") & data.index.normalize().isin(keep_days)]
    X_raw = train[features].to_numpy()
    scaler = _Scaler(X_raw)
    model = fit_hmm(scaler(X_raw), sequence_lengths(train.index, step),
                    n_states, n_iter, n_restarts, seed)
    target = future_target(data["label"], horizon).reindex(train.index).to_numpy()
    target_ok = data["split"].reindex(train.index + horizon).eq("train").to_numpy()
    post = filter_states(model, scaler(X_raw), _resets(train.index, step))
    target = np.where(target_ok, target, np.nan)
    rates = state_open_rates(post, target)
    return {
        "threshold": best_threshold(target, post @ rates) if tune_threshold else 0.5,
        "model": model, "scaler_mean": scaler.mean, "scaler_std": scaler.std,
        "state_open_rates": rates, "features": list(features), "step": step,
        "horizon_minutes": horizon_minutes, "trained_from": keep_days[0],
        "trained_to": keep_days[-1], "train_rows": len(train),
    }


def predict_open(bundle: dict, stream: pd.DataFrame) -> pd.Series:
    """Causal P(OPEN ``horizon_minutes`` ahead) for a stream of sensor rows."""
    X = (stream[bundle["features"]].to_numpy() - bundle["scaler_mean"]) / bundle["scaler_std"]
    post = filter_states(bundle["model"], X, _resets(stream.index, bundle["step"]))
    return pd.Series(post @ bundle["state_open_rates"], index=stream.index, name="p_open")


# ------------------------------------------------------------- metrics --

def evaluate(y_true: np.ndarray, p_open: np.ndarray, threshold=0.5, decision=None) -> dict:
    """Classification and probability scores.

    Decisions are ``p_open >= threshold`` (``threshold`` may be per row) unless
    ``decision`` gives them directly, e.g. from ``apply_hysteresis``.
    """
    y_true = np.asarray(y_true, dtype=int)
    p_open = np.asarray(p_open, dtype=float)
    if decision is None:
        y_pred = (p_open >= np.asarray(threshold, dtype=float)).astype(int)
    else:
        y_pred = np.asarray(decision, dtype=int)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "rows": len(y_true),
        "open_share": y_true.mean(),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if both_classes else np.nan,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, p_open),
        "roc_auc": roc_auc_score(y_true, p_open) if both_classes else np.nan,
    }


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) positions of the runs of True in a boolean array."""
    padded = np.r_[False, values, False].astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def action_episodes(
    predictions: pd.DataFrame, column: str = "p_open", threshold: float | str = 0.5,
) -> pd.DataFrame:
    """One row per period the umbrella should be OPEN, and per false opening.

    Works within each unbroken stretch of scored rows (an evaluation hour, or
    two adjacent ones). For every true OPEN period it records whether the
    prediction opened at all and what share of the period it covered; how late
    it opened if the period starts inside the stretch; and, if the period ends
    inside it, whether it had already closed too early or else how late it
    closed. Predicted OPEN periods that never overlap a true one are listed as
    false openings. ``threshold`` is a number or the name of a per-row
    threshold column.
    """
    step = predictions.index.to_series().diff().median()
    step_min = step / pd.Timedelta(minutes=1)
    rows = []
    for idx in np.split(np.arange(len(predictions)),
                        np.cumsum(sequence_lengths(predictions.index, step))[:-1]):
        chunk = predictions.iloc[idx]
        truth = chunk["target"].to_numpy() == 1
        limit = chunk[threshold].to_numpy() if isinstance(threshold, str) else threshold
        pred = chunk[column].to_numpy() >= limit
        for a, b in _runs(truth):
            opened = np.flatnonzero(pred[a:b])
            closed_after = np.flatnonzero(~pred[b:])
            rows.append({
                "kind": "should open", "start": chunk.index[a], "minutes": (b - a) * step_min,
                "reason": chunk["reason"].iloc[a],
                "opened": len(opened) > 0,
                # Only measurable when the true period starts inside the stretch.
                "open_delay_min": (opened[0] * step_min if len(opened) else np.nan)
                                  if a > 0 else np.nan,
                # Closing is judged only when the true period ends inside the stretch:
                # either it was already closed before the end, or it closed this late.
                "closed_early": (not pred[b - 1]) if b < len(truth) and len(opened) else np.nan,
                "close_delay_min": (closed_after[0] * step_min if len(closed_after) else np.nan)
                                   if b < len(truth) and pred[b - 1] else np.nan,
                "covered": pred[a:b].mean(),
            })
        for a, b in _runs(pred):
            if not truth[a:b].any():
                rows.append({"kind": "false opening", "start": chunk.index[a],
                             "minutes": (b - a) * step_min, "reason": ""})
    return pd.DataFrame(rows)


def evaluate_actions(
    predictions: pd.DataFrame, column: str = "p_open", threshold: float | str = 0.5,
) -> dict:
    """How well the umbrella opens and closes, from the user's point of view.

    ``threshold`` is a number or the name of a per-row threshold column.
    """
    y_true = predictions["target"].to_numpy() == 1
    limit = predictions[threshold].to_numpy() if isinstance(threshold, str) else threshold
    y_pred = predictions[column].to_numpy() >= limit
    episodes = action_episodes(predictions, column, threshold)
    should = episodes[episodes["kind"] == "should open"]
    false_open = episodes[episodes["kind"] == "false opening"]
    step = predictions.index.to_series().diff().median()
    hours = len(predictions) * step / pd.Timedelta(hours=1)
    breaks = np.r_[False, np.diff(predictions.index.as_unit("ns").asi8) != step.value]

    def switches(state: np.ndarray) -> int:
        return int(((np.diff(state.astype(int)) != 0) & ~breaks[1:]).sum())

    return {
        # Moment by moment
        "open when should be open": y_pred[y_true].mean(),
        "closed when should be closed": (~y_pred[~y_true]).mean(),
        "open when should be closed": y_pred[~y_true].mean(),
        "closed when should be open": (~y_pred[y_true]).mean(),
        # Per opening/closing
        "periods that should be open": len(should),
        "periods it opened for": should["opened"].mean() if len(should) else np.nan,
        "median open delay (min)": should["open_delay_min"].median(),
        "closed too early": should["closed_early"].astype(float).mean(),
        "median close delay (min)": should["close_delay_min"].median(),
        "share of open period covered": should["covered"].mean() if len(should) else np.nan,
        "false openings per 24 h": len(false_open) / hours * 24,
        "switches per 24 h": switches(y_pred) / hours * 24,
        "label switches per 24 h": switches(y_true) / hours * 24,
    }

