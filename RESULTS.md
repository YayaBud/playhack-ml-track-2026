# Results

Out-of-fold over 3000 training athletes, 3 seeds x 5 folds, with feature
selection performed inside each fold. `submission.csv` comes from the same
models refit on all training data (`src/build_final_models.py`).

## Scored metric (problem statement, page 6)

`skill = max(0, 1 - MAE_model / MAE_baseline)`, where a **missed** injury
contributes a flat `PENALTY = 30` to both timing heads, and the baseline
predicts the training-set mean for every injured athlete. Task B is
evaluated over the truly-injured athletes only. See `src/score.py`.

| | scored quantity | **ours** | ceiling | closed |
|---|---|---|---|---|
| **Task A** | F1 | **0.5210** | 0.5246 | 99.3% |
| **Task B** | skill — `onset_day_offset` | **0.6541** | 0.6647 | 98.4% |
| **Task B** | skill — `recovery_duration` | **0.1075** | 0.1329 | 80.9% |
|  | balanced mean of the three | **0.4275** | 0.4407 | 97.0% |

Underlying errors: onset MAE **2.6341** against a 7.6148 baseline; recovery
MAE **2.8930** against 3.2416. OOF recall on injured athletes: **1.0000**.

### Why the threshold sits so low

A miss costs 30 against baselines of ~7.6 and ~3.2, so break-even recall
is 0.82 (onset) and 0.99 (recovery). Tuning for F1 alone clears neither:

| threshold | F1 | skill_onset | skill_recovery | mean |
|---|---|---|---|---|
| 0.38 — F1-optimal | **0.6346** | 0.0000 | 0.0000 | 0.2115 |
| 0.045 — **shipped** | 0.5210 | **0.6541** | **0.1075** | **0.4275** |

The shipped rule flags every athlete except the most confidently healthy
0.63% (half the maximum exclusion that still holds OOF recall at 1.0).
Test positive rate 0.9936. A perfectly-calibrated oracle makes the same
trade, so this is the optimal play under the rule, not a quirk of our model.

## Diagnostics (not scored)

| head | metric | model | ceiling | closed |
|---|---|---|---|---|
| `injured_in_risk_window` | ROC-AUC | 0.7625 | 0.7702 | 97.1% |
| `onset_day_offset` | MAE (hits only) | 2.6341 | 2.5533 | 98.4% |
| `recovery_duration` | MAE (hits only) | 2.8926 | 2.8543 | 89.9% |

> A ceiling is an **estimate** of irreducible error, not a proven bound. It
> must condition on at least as much information as the model it bounds -- an
> earlier estimate built on a single recovered feature put this model at 113%
> of "maximum", which signalled a bad estimate rather than a good model.
> `src/official_ceiling.py` now warns whenever a score exceeds its ceiling.

## Per-model out-of-fold scores

| head | LightGBM | XGBoost | CatBoost | group median | blend |
|---|---|---|---|---|---|
| `injured_in_risk_window` (AUC) | 0.7609 | 0.7623 | 0.7591 | - | **0.7625** |
| `onset_day_offset` (MAE) | 2.6391 | 2.6393 | 2.6782 | - | **2.6341** |
| `recovery_duration` (MAE) | 2.9313 | 2.9570 | 2.9154 | 2.8981 | **2.8926** |

`recovery_duration` carries a fourth candidate: a cross-fitted sport+gender
median. It beats all three tree models on its own and takes the largest
share of the blend -- recovery has little per-athlete signal beyond sport.

- classifier: {"lgb": 0.04, "xgb": 0.826, "cat": 0.134}
- `onset_day_offset`: {"lgb": 0.595, "xgb": 0.405, "cat": 0.0}
- `recovery_duration`: {"lgb": 0.204, "xgb": 0.0, "cat": 0.216, "grp": 0.579}
