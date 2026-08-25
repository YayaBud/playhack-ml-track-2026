# Results

Out-of-fold, 3-seed repeated 5-fold CV, features selected inside each
fold. `submission.csv` is produced by the same run.

## Model vs Bayes ceiling

| head | metric | baseline | **model** | ceiling | gap closed |
|---|---|---|---|---|---|
| `injured_in_risk_window` | ROC-AUC | 0.5000 | **0.7718** | 0.7727 | 99.7% |
| `injured_in_risk_window` | F1 | - | **0.6396** | 0.6404 | 99.9% of ceiling |
| `onset_day_offset` | MAE | 7.6086 | **2.6206** | 2.5152 | 97.9% |
| `recovery_duration` | MAE | 3.2333 | **2.8690** | 2.8410 | 92.9% |

## SOTA benchmark: ours vs AutoGluon (best_quality)

Same features (`src/features.py`), same train/test split.
AutoGluon 1.6.1, `presets="best_quality"`, 300s/head, 5-fold
internal bagging -- a standard, widely-cited AutoML SOTA
reference (stacked LightGBM/XGBoost/CatBoost/RF/ExtraTrees/NN/KNN,
weighted-ensembled). TabPFN (pretrained tabular transformer,
also SOTA for this data regime) was blocked by a one-time
license click-through its pip package now requires -- not
something completable headlessly; see `src/bench_tabpfn.py`.

| head | metric | ours (solo) | AutoGluon (solo) | **final (best/blend)** |
|---|---|---|---|---|
| `injured_in_risk_window` | AUC | 0.7636 | 0.7718 | **0.7718** (blend, 95% AG) |
| `onset_day_offset` | MAE | 2.6227 | 2.6466 | **2.6206** (blend) |
| `recovery_duration` | MAE | 2.8950 | 2.8690 | **2.8690** (AutoGluon) |

AutoGluon wins classifier and recovery outright; our L1-objective,
in-fold-selected regressor wins onset outright, but a 78/22 blend
with AutoGluon's onset model still edges out both solo scores --
the two approaches make different enough errors that averaging
them helps. `submission.csv` ships the best/blended choice per head.

Baselines: random ranking for AUC, constant-median prediction for MAE.
Ceilings: analytic expected AUC of the Bayes ranking; conditional
mean-absolute-deviation for the MAE heads. Lower MAE is better, so
"gap closed" moves from the constant baseline down toward the ceiling.

## Per-model out-of-fold scores

| head | LightGBM | XGBoost | CatBoost | sport+gender median | blend |
|---|---|---|---|---|---|
| `injured_in_risk_window` (AUC) | 0.7601 | 0.7633 | 0.7621 | - | **0.7636** |
| `onset_day_offset` (MAE) | 2.6326 | 2.6264 | 2.6555 | - | **2.6227** |
| `recovery_duration` (MAE) | 2.9428 | 2.9478 | 2.9283 | 2.8981 | **2.8950** |

`recovery_duration` folds in a 4th candidate: a cross-fitted
sport+gender group median. It beats all three tree models
individually (2.898 vs 2.928-2.948) and takes 62% of the blend
weight -- confirming the finding in docs/findings.md that
recovery has almost no per-athlete signal beyond sport identity.

Blend weights: classifier {"lgb": 0.002, "xgb": 0.546, "cat": 0.452}
- `onset_day_offset`: {"lgb": 0.359, "xgb": 0.641, "cat": 0.0}
- `recovery_duration`: {"lgb": 0.18, "xgb": 0.0, "cat": 0.205, "grp": 0.615}

## Submission

- decision threshold 0.290 on isotonic-calibrated probabilities, chosen to maximise expected F1 on test's own calibrated scores
- predicted positive rate 0.2382 vs train 0.3500
- test's calibrated probabilities average 0.3853 (a prevalence estimate consistent with the analytic one below) but are sharply bimodal: ~78% of athletes cluster at p~0.15-0.30 (background hazard) and ~23% at p>=0.65 (overload hazard), with almost nothing in between. The F1-optimal threshold sits in that gap and isolates the overload cluster -- matching the mean (0.385) instead would flag every background-hazard athlete too and tank precision for a small recall gain
- overload prevalence is the real driver of the test/train gap: 22.0% of test athletes sit above the overload ACWR threshold vs 17.5% in train (rates conditional on overload/background are ~0.93 / ~0.23 in both splits, so this is a covariate shift, not a labelling shift) -- an independent analytic estimate from that decomposition lands at 0.371-0.385 across four ACWR proxies, matching the model's own calibrated mean
- 206 engineered features available; 35 / 35 / 20 selected per head
