# Automatic Umbrella

Decides whether an automatic umbrella should be **OPEN** or **CLOSED**, with a probability, from simple weather sensors.
It opens for **rain** and for **heat stress** (UTCI, via `pythermalcomfort`).

The pipeline turns 1-minute weather-station data into 20-second data, labels it, and trains a Gaussian HMM that is evaluated the way it would run on the device.

```text
dataset/  (KMA AWS, 1 min)
   │  simulation.ipynb   1 min → 20 s (PCHIP)
   ▼
simulated_dataset/  (20 s, one CSV per month)
   │  labeling.ipynb     UTCI + rain → p_open, label OPEN/CLOSED
   ▼
labeled_dataset/  (20 s + labels)
   │  training.ipynb     window features → split → walk-forward HMM → evaluation
   ▼
models/umbrella_hmm.joblib
```

## Setup

```bash
conda create -n weather_env python=3.9
conda activate weather_env
pip install -r deployment/requirements.txt
python -m ipykernel install --user --name weather_env --display-name "Python (weather_env)"
```

On Linux and the Jetson Orin Nano (connection, Wi-Fi, venv or conda): see [deployment/SETUP.md](deployment/SETUP.md).

Run the notebooks in order with the **weather_env** kernel. Restart the kernel after changing anything in `modules/`.

## Data

`dataset/` holds the raw minute data from **KMA automatic weather station 939** (cp949 CSVs, Korean headers; git-ignored).
It covers 2024-01-01 → 2026-08-09, **without 2025**, and includes about 70 gaps; the longest is a 13.6-hour outage on 2024-07-22.
Timestamps label the **end** of each minute.

| Column | Meaning | Used for |
|---|---|---|
| `temperature`, `humidity`, `wind_speed` | air temperature (°C), relative humidity (%), wind speed (m/s) | UTCI and model features |
| `rain_flag` | rain detector, 0 = dry / 10 = wet | rain labels and features |
| `rain` | rain gauge, mm per minute (0.5 mm steps) | rain labels only |
| `wind_dir` | wind direction (°) | available, not used yet |
| `pressure`, `sea_pressure` | barometric pressure | not used (no sensor on the device) |
| `solar`, `sunshine` | placeholders only (−9.97, 64539) | dropped |

The model only uses sensors the device has: **temperature, humidity, wind, rain detector, clock**.

## 1. Simulation: `simulation.ipynb`

Upsamples the 1-minute data to 20 s with `modules/simulation/pchip.py`; `modules/dataset/dataset.py` adapts the raw files.

- **Modes:** each variable is treated according to how it was measured:

  | Mode | Variables | Rule |
  |---|---|---|
  | `point` | temperature, pressure, humidity | PCHIP curve through every minute value |
  | `mean` | wind speed, wind direction (as sin/cos) | 20 s values average back to the minute value |
  | `total` | rain | minute amount split evenly over the minute |

- **Gaps:** each variable is interpolated within its own unbroken runs and never across a gap.
- **Missing values:** missing rain becomes 0; rows still missing other values are dropped.
- **Rain detector:** `rain_flag` is held constant across each minute.
- **Settings:** `START`, `END`, `STEP_SECONDS` (20).
- **Output:** `simulated_dataset/<YYYY-MM>.csv`.

The 20 s data is a deterministic interpolation, not measured 20 s truth.

## 2. Labeling: `labeling.ipynb`

Adds the target `label` to every 20 s row:

- **Heat:**
  - `utci` comes from `pythermalcomfort.models.utci` (temperature, humidity, wind clipped to 0.5–17 m/s).
  - Mean radiant temperature = air temperature (shade), because the station has no solar data.
  - `p_heat` is a logistic curve on the 10-minute mean UTCI: 5 % at 26 °C, 50 % at 29 °C, 95 % at 32 °C.
  - Heat only counts 07:00–19:00.
- **Rain:** `p_rain` is 1 while the gauge or detector is wet, then fades as `exp(−minutes / 5)`.
- **Combined:** `p_open = 1 − (1 − p_heat)(1 − p_rain)`; `open_percent` = `p_open` × 100.
- **Label:** `OPEN` at `p_open ≥ 0.5`, back to `CLOSED` only below 0.3 (hysteresis, no flapping).
  - `reason` = `heat`, `rain` or `heat+rain`.
- **Output:** `labeled_dataset/<YYYY-MM>.csv` (simulated columns + `utci`, `stress_category`, `p_heat`, `p_rain`, `p_open`, `open_percent`, `label`, `reason`).

## 3. Training: `training.ipynb`

1. **Load:** `load_simulated()` reads `simulated_dataset/` and adds `segment`, which numbers unbroken stretches. Sequences never cross a gap.
2. **Split:** `hourly_split()`: every day, `N_HOURS` (3) random hours from 06:00 onward become evaluation.
   - One hour is drawn from each part of the day (06–11, 12–17, 18–23).
   - A 30-minute buffer around each evaluation hour belongs to neither set, which prevents leakage.
3. **Labels:** joins `label`, `reason` and `utci` from `labeled_dataset/`.
4. **Window features:** `add_window_features()`, still at 20 s.
   - Each row summarises the **past** few minutes; windows never cross gaps.
   - Computed on the whole stream before splitting.

   | Feature | From |
   |---|---|
   | `utci_mean_10m`, `utci_max_10m` | UTCI (temperature, humidity, wind) |
   | `wet_share_5m`, `wet_share_10m`, `rain_recency` | rain detector |
   | `humidity`, `humidity_chg_15m` | humidity |
   | `wind_mean_5m`, `wind_max_5m` | wind speed |
   | `hour_sin`, `hour_cos` | clock |

5. **Walk-forward evaluation (deployment simulation):** `walk_forward()`.
   - Retrains every 7 days on the training rows of the last `WINDOW_DAYS` (60) days.
   - The first fit keeps the best of several random starts; later fits warm-start from the previous week.
   - If a fit fails, it refits from scratch, and otherwise keeps the previous model.
   - The HMM (`RidgeGaussianHMM`, full covariance with a ridge so covariances stay valid) reads the sensor stream **forward only**.
   - Each hidden state gets an OPEN rate from the training labels, and `P(OPEN in HORIZON_MINUTES)` = Σ P(state) × rate.
   - Decision: per-fold F1-tuned threshold, then hysteresis (`CLOSE_MARGIN` 0.2) and a `MIN_HOLD_MINUTES` (3) hold time.
6. **Evaluation:** compared with **persistence**, i.e. "the label stays as it is now".
   - Row metrics: F1, precision, recall, ROC-AUC, Brier.
   - Recall per reason.
   - **Open/close behaviour:** periods opened for, opening/closing delay, closed too early, false openings and switches per 24 h.
7. **Final model:** `train_final()` fits on the latest window and saves `models/umbrella_hmm.joblib` (HMM, scaler, state OPEN rates, threshold, close margin, hold time).

Main settings (`training.ipynb`, section 4):

| Setting | Default | Meaning |
|---|---|---|
| `N_STATES` | 10 | hidden states (5 trains faster, weaker on heat) |
| `HORIZON_MINUTES` | 5 | predict the label this far ahead |
| `RETRAIN_DAYS` / `WINDOW_DAYS` | 7 / 60 | retrain weekly on the latest 60 days of data |
| `USE_WINDOWS` | True | window features (False = current readings only) |
| `TUNE_THRESHOLD` | True | F1-optimal threshold per retrain |
| `CLOSE_MARGIN`, `MIN_HOLD_MINUTES` | 0.2, 3 | anti-flapping rules for the decision |

A full 2024–2026 run with 10 states takes a few hours.

## Deployment

On the device, every 20 s:

```text
sensor reading → UTCI (pythermalcomfort) → window features (last 15 min buffer)
              → HMM forward step → p_open → threshold + hysteresis + hold → OPEN / CLOSED
```

`predict_open(bundle, stream)` in `modules/model/hmm.py` gives `p_open` for a stream of rows with the model's features.

## Current status

Walk-forward on March–September 2024 (10 states, window features, device sensors only):
- F1 ≈ 0.69; rain recall ≈ 95 %, heat recall ≈ 53 %.
- ≈ 3.6 switches/day; rarely closes too early; closes ≈ 10 min late after rain.
- **Persistence still scores higher (F1 ≈ 0.97).**

The labels are computed from current sensors, so a rule on live sensors is a strong baseline.
Next step under consideration: a small classifier on top of the HMM state probabilities, mainly to improve heat.

Limitations:
- The 20 s data is interpolated.
- UTCI assumes shade.
- The labels are rule-based, not observed.
- There is no 2025 data.

## Repository layout

```text
simulation.ipynb, labeling.ipynb, training.ipynb   pipeline notebooks (run in this order)
modules/simulation/pchip.py       1 min → 20 s upsampling (point / mean / total)
modules/dataset/dataset.py        load_aws, simulate, load_simulated
modules/dataset/split.py          hourly_split, sequence_lengths
modules/dataset/windowing.py      add_window_features, WINDOW_FEATURES
modules/model/hmm.py              RidgeGaussianHMM, walk_forward, evaluate, evaluate_actions, train_final, predict_open
dataset/, simulated_dataset/, labeled_dataset/, models/   data and outputs
deployment/                       SETUP.md (Jetson + environment), README.md (script usage), runtime modules/ (dataset, inference), run_simulation.py, test_shelly.py (+ modules/fake_shelly.py), models/umbrella_hmm.npz,
                                  extract_evaluation.py, evaluation_dataset/ (held-out clips)
legacy/, requirements_legacy.txt  previous 1-minute HMM prototype
```
