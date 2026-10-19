# Review of the real-time pipeline (`ee5f7d3`)

Dimas Maulana Fasya · 19 October 2026
Working branch: `dfasya/prototype-automatic-umbrella` → `dimas-improvements`

I read the new pipeline before building the canopy actuation layer on top of it.
This is what I found. The code review is in sections 2 and 5; sections 3 and 4
are arguments, not corrections, and I would rather be told I am wrong about them
than have them quietly adopted.

What I worked from: `README.md`, `modules/simulation/pchip.py`,
`modules/model/hmm.py`, `modules/dataset/{dataset,windowing,split}.py`,
`models/umbrella_hmm.joblib`, and `simulated_dataset/`. I have not seen the
notebooks, so anything that depends on them is marked as a question.

---

## 1. What holds up

Worth saying first, because most of it does.

**Causality is handled properly.** `walk_forward` fits on days
`[start − window_days, start)` and scores `[start, start + retrain_days)`. The
forward filter is genuinely forward-only. There is no look-ahead anywhere I
could find.

**`train_target_ok` is a real catch.** Restricting the rows used for
`state_open_rates` to those whose *future* label row is also a training row
closes a leak that most walk-forward code leaves open. It is easy to miss.

**The window length is bounded by the split gap.** `MAX_WINDOW` is 15 minutes
and the split buffer is 30, so a training row's history never reaches into an
evaluation hour. The docstring states the constraint explicitly, which is how it
should be done.

**The 0/10 rain encoding is handled correctly.** `wet = (df[wet_columns] > 0)`
rather than an equality against 1. This is the detail most likely to break a
sensor rebuild later, and it is right here.

**`RidgeGaussianHMM` is a correct diagnosis.** `hmmlearn` applies `min_covar`
only at initialisation, so a feature that is constant within a state collapses
that state's covariance. Adding the ridge after each M-step is the right fix.

**`pchip.py` is unusually careful.** Mean conservation is asserted at
`rtol=1e-12` and raises rather than warns; interpolation never crosses a gap;
the `attrs` block states that the output is not measured truth. The
bound-respecting shrinkage that preserves minute means, rather than clipping the
finished output, is a nice piece of work.

**Heat is gated to 07:00–19:00.** This fixes the night-firing behaviour the old
`heat_score` had.

**The README says persistence wins.** Reporting a baseline that beats your own
model is the part of this that I trust most.

---

## 2. Blocking: the repository cannot be reproduced from a clone

This one needs fixing before anything else, because it stops the pipeline from
running at all on a fresh clone.

`simulated_dataset/*.csv` does not contain `rain_flag`. Checked directly:

```
simulated_dataset/2026-07.csv
columns: timestamp, temperature, pressure, sea_pressure, humidity,
         wind_speed, rain, wind_dir, wind_u, wind_v
rows:    30231      2026-07-01 00:00:00 → 2026-07-07 23:58:40
```

`load_simulated()` knows this can happen and patches it from the raw files
(`dataset.py`, lines 130–133):

```python
if "rain_flag" not in df and raw_source is not None:
    raw = load_aws(raw_source).set_index("timestamp")
    ...
```

But `dataset/` is git-ignored, so on a clone `load_aws("dataset")` raises
`FileNotFoundError: No CSV files found` at line 77. Both `labeling.ipynb` and
`training.ipynb` go through `load_simulated`, so **neither notebook runs on a
clean clone.** Passing `raw_source=None` does not help: `add_window_features`
then fails on its default `wet_columns=("rain_flag",)`, and substituting the
`rain` gauge silently changes the instrument behind `wet_share_5m`,
`wet_share_10m` and `rain_recency`.

The shape of this is worth noting on its own: 182 MiB of *derived* CSVs are
committed to git forever, while the one small column that **cannot** be
regenerated without the raw files was left out. The ignore policy is inverted —
the reproducible artefacts are in the repository and the irreplaceable input is
not.

**Fix, and it is small.** `simulate()` already has `hold=("rain_flag",)` in its
defaults and writes that column at lines 245–247, so the column is probably
being dropped by a column selection in `simulation.ipynb` before saving. Writing
it out adds one integer column to files that already exist, and after that
`simulated_dataset/` stands on its own and nothing has to be sent to anyone.

**Until then I need the `dataset/` folder** — the same KMA station 939 one-minute
CSVs, nothing new. Roughly 80–100 MB, much less compressed.

---

## 3. Three findings about the deployed model

### 3.1 The published numbers do not describe the committed model

From `models/umbrella_hmm.joblib`:

| | |
|---|---|
| features | the 11 window features |
| states | 10 |
| threshold | 0.70 |
| horizon | 5 min |
| trained on | 2026-05-09 → 2026-07-07, 195,305 rows |

The README reports F1 ≈ 0.69 from a walk-forward over **March–September 2024**.
The bundle was trained on **May–July 2026**. Different year, different season,
different data.

So there is no published performance figure for the model that is actually in
the repository. That matters to me immediately, because I am writing progress
reports and I do not want to quote 0.69 as this system's performance if it
describes a different run. Has a 2026 walk-forward been done?

Two things did check out against the README: the feature list and state count
match exactly, and 195,305 rows is what a 60-day window gives after
`hourly_split` removes three evaluation hours and their buffers
(60 × 1440 × 3 × 0.75 ≈ 194,400). The threshold of 0.70 means
`TUNE_THRESHOLD=True` was used; with `CLOSE_MARGIN` 0.2 the device opens at
p ≥ 0.70 and closes below 0.50.

### 3.2 A month of data never entered the pipeline

`simulated_dataset/2026-07.csv` ends at **7 July 2026**, and the bundle's
`trained_to` agrees. The README says the raw data runs to **9 August 2026**.

So 8 July – 9 August exists in `dataset/` but was never simulated, labelled or
trained on. That month is the peak of the jangma and the wettest stretch of the
year in southern Korea. For a device whose primary trigger is rain, leaving out
the heaviest rain period is a real gap — and the deployed model's 60-day window
ends immediately before it.

A related consequence: the model has seen May, June and early July only. If the
device runs this autumn it is operating on conditions it has never been trained
on.

### 3.3 Wind can only open the canopy, never close it

This is the one I would fix first.

Wind enters the decision only through UTCI, where it *cools*. Higher wind lowers
`p_heat`, so wind makes a heat opening **less** likely — which looks protective
but is a side effect, not a rule.

For rain it does nothing. `p_rain` is 1 whenever the detector is wet, regardless
of wind speed. A thunderstorm is wet and windy at the same time, so the current
logic extends the fabric into exactly the conditions that damage it.

The data is already there. `wind_mean_5m` and `wind_max_5m` are computed and fed
to the model. What is missing is a rule that says: above a threshold, close,
whatever the model says.

For thresholds, EN 13561 (external blinds and awnings) wind resistance classes
have published sensor setting values:

| Class | Setting |
|---|---|
| 1 | 7.8 m/s (28 km/h) |
| 2 | 10.6 m/s (38 km/h) |
| 3 | 13.3 m/s (48 km/h) |

For comparison, the KMA strong-wind advisory for land is 14 m/s sustained, which
is above every class in that table. By the time an advisory exists, a class 3
awning is already past its limit — so weather warnings are not a usable trigger
here. The threshold has to be local and considerably lower.

I have written `modules/model/wind_safety.py` on my branch. It sits after
`apply_hysteresis`, can only force CLOSED and never open, latches at the
threshold, releases only after the gust has stayed low for a lockout period, and
returns a mask of the rows it changed so the cost can be measured rather than
assumed. It does not modify `hmm.py`. Default is class 1, the most conservative,
until the canopy's actual rating is known.

Two honest limits on it: wind at station 939 is not wind at the canopy, so a
local anemometer is the only real input; and no sensor catches a gust that has
already arrived.

---

## 4. Two methodological arguments

These are arguments. I may be wrong about either.

### 4.1 Mean radiant temperature is set to air temperature, for a device whose job is shade

The labels use `MRT = air temperature` because the station has no usable solar
data. That makes the UTCI in the label a **shade** UTCI.

The problem is that shading is the whole function of the canopy. With
MRT = T<sub>air</sub>, the label cannot distinguish 29 °C in full sun from 29 °C
under cloud, and those differ by something like 8–12 K in MRT. The heat trigger
is being trained against a target with its dominant driver removed. Heat recall
of 53 % is consistent with that: there is not much signal left beyond temperature
itself.

There is also a circularity. The system decides to deploy shade based on a
comfort index that already assumes shade exists. The decision variable and the
post-decision state are the same quantity.

Three ways out, increasing in cost:

1. Estimate MRT from clear-sky solar geometry (`pvlib` solar position plus a
   clear-sky model) and a cloud proxy, and compute a *sun-exposed* MRT for the
   canopy-closed state. Approximate, but it restores the variable that actually
   drives the decision. I can prototype this — it needs `simulated_dataset/`
   only, once `rain_flag` is available.
2. Install a globe thermometer or pyranometer at the canopy and label from
   measured MRT. This is already on our sensor list.
3. At minimum, state in any report that heat performance assumes shade MRT, and
   is therefore a lower bound on what a real device could do.

### 4.2 The persistence gap is a property of the label, not of the model

The README notes that persistence scores F1 ≈ 0.97 against the HMM's 0.69, and
that "a rule on live sensors is a strong baseline". I think the conclusion is
stronger than that.

`p_open` is a deterministic function of UTCI and the rain detector, both read
from the current sensors. The model's features are those same sensors. The
target is the same function five minutes later. So:

- The label is close to piecewise constant by construction. Hysteresis at
  0.5/0.3 plus a 3-minute hold gives long plateaus. Over 20-second rows,
  "unchanged" is correct almost always, and row-level F1 is dominated by the
  quiet stretches. It says almost nothing about the transitions, which are the
  only part that matters.
- The HMM compresses 11 continuous features into a 10-way soft assignment and
  then a single scalar through `post @ rates`. Ten states cannot represent a
  smooth logistic in UTCI. The label is analytically available from the
  features, so the bottleneck guarantees a loss against anything reading the
  features directly.

A classifier stacked on the state probabilities keeps the bottleneck and then
tries to recover from its output. If a classifier is worth fitting, fit it on
the features and add the state probabilities as extra inputs only if they earn
their place.

The two coherent directions as I see them:

- **Change the target.** Predict something the current sensors do not determine
  — rain onset at +15 or +30 minutes — and score only transition windows. Then
  the forecasting framing means something.
- **Change the model.** If the target stays at +5 minutes of a current-sensor
  rule, a direct supervised model will win, and the honest finding is that the
  HMM adds nothing here.

One detail that explains something you already noticed: because `rates` are
trained against the *shifted* label, the horizon is absorbed into one static
number per state. The transition matrix then only affects filtering, not the
forecast, so a state cannot distinguish "heading into rain" from "heading out of
it". More states partially compensate. That is probably why 10 beat 5 on heat.

---

## 5. Small fixes

**`evaluate_actions` can silently report zero switches.** `hmm.py` line 506:

```python
breaks = np.r_[False, np.diff(predictions.index.asi8) != step.value]
```

`DatetimeIndex.asi8` returns the index's own resolution, which is not always
nanoseconds, while `step.value` is always nanoseconds. When they disagree, every
row is marked as a break and `switches()` returns 0 — no error, just a wrong
metric. I hit this in my own code on pandas 3.0, where `date_range` gives
microsecond resolution. Comparing Timedeltas instead is resolution-safe:

```python
breaks = (predictions.index.to_series().diff() != step).to_numpy()
```

The same pattern is in `apply_hysteresis` via `sequence_lengths`, worth a check.

**`pchip.py` hardcodes the wrong step in its metadata.** The `attrs` block says
`"not measured 5s truth"` whatever `step_seconds` is. At 20 s the metadata is
wrong. One line.

**Module defaults disagree with the README.** Four of them:

| | `hmm.py` | README |
|---|---|---|
| `FEATURES` | 7 current readings | 11 window features |
| `n_states` | 5 | 10 |
| `window_days` | 365 | 60 |
| `tune_threshold` | False | True |

The notebooks clearly pass the right values — the bundle proves it. But anyone
calling `walk_forward()` or `train_final()` directly gets the weaker
configuration with no warning.

**`_check_healthy` may not catch what its message claims.** It raises
`"a state is never used"` when `transmat_.sum(axis=1)` is not 1. An unused state
usually shows as a near-zero *column* with rows still summing to 1, so the check
probably does not detect that case.

**Threshold is tuned on the wrong objective.** `best_threshold` maximises
row-level F1 — correctly, on training rows only. But `evaluate_actions` already
computes the quantities that matter operationally: minutes late to open, false
openings per 24 h, switches per day. Tuning against one of those would be a
cheap improvement.

**Reported counts are inflated threefold.** Every real minute contributes three
20-second rows that are a deterministic function of the same observation. The
point estimates are unaffected, but "4.7 million rows" is really ~1.6 million
minutes, and any confidence interval implied by *n* is too narrow. The
upsampling gives no predictive benefit — its purpose is to make the training
data the same shape as the device stream so one code path serves both, which is
a good reason. It is worth stating as an engineering convenience rather than as
data.

---

## 6. What I am asking for

1. **`dataset/`** — the raw station 939 CSVs. Everything downstream is blocked
   without them.
2. **Write `rain_flag` into `simulated_dataset/`** so this does not recur.
3. **Is there a walk-forward run for 2026?** If not, the README's numbers and
   the committed bundle should not be quoted together.
4. **Was 8 July – 9 August left out deliberately?**
5. **Reactions to sections 4.1 and 4.2**, before I build anything on either.

What I am doing meanwhile, on my own branch, touching nothing of yours:
`modules/model/wind_safety.py`, and the actuator layer (Shelly 2PM Gen4 driver,
fan staging) that sits on top of `predict_open`.

---

## 7. References

- EN 13561 wind resistance classes, manufacturer sensor setting values:
  [Griesser wind fact sheet, awnings](https://www.griesser.com/ch/wp-content/uploads/2026/03/Wind-fact-sheet-IT_Awnings.pdf)
- KMA strong wind advisory and warning thresholds:
  [Hanam City, KMA warning criteria](https://www.hanam.go.kr/www/contents.do?key=3303)
- UTCI: `pythermalcomfort.models.utci`, valid for wind 0.5–17 m/s
- Upstream repository: `Sekigahara/prototype-automatic-umbrella` at `ee5f7d3`

Model and data figures in sections 2 and 3.1 were read directly from
`models/umbrella_hmm.joblib` and `simulated_dataset/2026-07.csv` in a clone of
`ee5f7d3`. Figures attributed to the README are quoted from it and not
independently reproduced.
