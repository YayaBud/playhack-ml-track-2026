================================================================================
PLAYHACK 2026 - ML TRACK  |  ROUND 1 SUBMISSION
Athlete Injury Risk Prediction
IIT Guwahati  |  Sports Board x Technical Board
================================================================================

WHAT THIS IS
--------------------------------------------------------------------------------
Given a 30-day observation window of wearable + training data per athlete, this
predicts what happens in the following 30-day risk window:

  injured_in_risk_window   0/1   does an injury onset occur          (Task A)
  onset_day_offset         1-30  which day of the risk window        (Task B)
  recovery_duration        days  how long the athlete is sidelined   (Task B)

submission.csv (included) holds predictions for all 1,100 test athletes.
Nothing needs to be retrained to reproduce it.


================================================================================
QUICK START - REPRODUCE submission.csv IN TWO COMMANDS
================================================================================

  1) Install dependencies (Python 3.12 recommended):

       pip install -r requirements.txt

  2) Run inference, pointing --data-root at the folder that CONTAINS the
     "test" folder of competition CSVs:

       python predict.py --data-root /path/to/data --split test --out submission.csv

     Windows example:
       python predict.py --data-root C:\playhack\data --split test

Expected console output (takes ~1-2 minutes, mostly feature building):

       building features from ...\data\test ...
         [leak-check] test: sources clipped to 2026-01-05..2026-02-03
         [test] +daily_features       -> 94 cols
         ... (5 more feature groups) ...
       wrote submission.csv  rows=1100  positive_rate=0.9936

The output is byte-identical to the bundled submission.csv (we verified this by
extracting this ZIP into a clean directory and diffing).

NOTE: the competition CSVs are NOT bundled (they are ~660 MB and you already
have them). Only the "test" split is required - no training data is needed for
inference. If --data-root is wrong, the script exits with instructions rather
than a stack trace.


================================================================================
HOW TO VERIFY OUR CLAIMS
================================================================================

  python src/validate_submission.py     Checks submission.csv against
                                        sample_submission.csv: column names and
                                        order, row count, dtypes, value ranges,
                                        and that no field is blank.

  python src/score.py                   Self-check of our implementation of the
                                        official metric (5 assertions, incl. the
                                        penalty rule and the baseline scoring 0).

  python src/test_leakage.py            Proves no feature can see the risk
                                        window, including a negative control
                                        that must fail if the guard is removed.

  python src/threshold_sweep.py         Regenerates the full 102-threshold
                                        sweep behind our decision threshold.
                                        (Needs the training labels.)

Raw numbers behind every claim: reports/metrics.json, reports/threshold_sweep.csv


================================================================================
WHAT IS IN THIS ARCHIVE
================================================================================

  predict.py               Inference entry point (raw CSVs -> submission.csv)
  submission.csv           Final predictions, 1,100 test athletes
  requirements.txt         Pinned dependencies
  readme.txt               This file
  RUNME.md / README.md     Same guidance in Markdown, plus results
  RESULTS.md               Scores, per-model breakdown, benchmark

  models/                  The trained models (2.2 MB total)
    manifest.json            Feature lists per head, blend weights, decision
                             threshold, category orderings, library versions
    clf_{lgb,xgb,cat}.joblib Injury classifier ensemble
    onset_{lgb,xgb}.joblib   Onset-day regressors
    recovery_*.joblib        Recovery regressors + sport/gender median table

  src/                     All code (feature engineering, training, scoring,
                           evaluation, benchmarks, tests)
  reports/                 Figures (EDA + threshold analysis), metrics, sweep CSV
  docs/                    findings.md, methodology.md, problem_statement.md


================================================================================
MODEL SUMMARY
================================================================================

Features    223 engineered per athlete, built ONLY from observation-window days
            (2026-01-05 .. 2026-02-03) for both train and test. Core construct
            is ACWR (acute:chronic workload ratio) plus monotony, strain, load
            trend, resting-HR drift, sleep debt and training-gap structure.
            20-35 features selected per head, inside each CV fold.

Models      LightGBM + XGBoost + CatBoost, blended with non-negative weights
            fitted on out-of-fold predictions. 3 seeds x 5 folds. L1 objectives
            for the two day-count heads (MAE targets the conditional median).
            recovery_duration additionally blends a cross-fitted sport+gender
            median, which alone beats all three tree models.

Threshold   0.046 (rank-quantile form: flag all but the most confidently healthy
            ~0.6%). Chosen against the official metric, not F1 - see below.

Out-of-fold results (3,000 training athletes):
            Task A  F1                     0.5210
            Task B  onset skill            0.6541
            Task B  recovery skill         0.1075
                    balanced mean          0.4275   (97% of estimated ceiling)
            Diagnostic only: ROC-AUC       0.7625


================================================================================
THE KEY DESIGN DECISION (why we flag ~99% of athletes)
================================================================================

Per the problem statement (page 6), Task B is scored ONLY over athletes who are
truly injured, and a MISSED injury costs a flat penalty of 30 on both timing
heads - against baselines of just 7.61 (onset) and 3.24 (recovery). A false
positive is not in that population at all, so it costs Task B nothing.

One miss therefore wipes out roughly four good onset predictions or nine
recovery ones. Break-even recall for any nonzero skill is 0.82 (onset) and
0.99 (recovery).

We tested this rather than assuming it: a full out-of-fold sweep over 102
thresholds (src/threshold_sweep.py, raw table in reports/threshold_sweep.csv).

    threshold    flagged   missed   F1      skill_on  skill_rc
    0.046 (ours)  99.4%       0     0.5210   0.6541    0.1075
    0.10          94.9%      22     0.5276   0.5833    0.0000
    0.38 (F1-opt) 22.6%     502     0.6346   0.0000    0.0000

Tuning for F1 - the obvious default - scores ZERO on two of the three
components. Because the final Task A / Task B weighting is not published, we
ran a sensitivity analysis over seven weightings: our threshold ranks 3rd or
4th of 102 in six of them, within 0.001 of the best possible score. It is only
suboptimal if Task A carries 80%+ of the final rank.

F1 of 0.52 is not a weak classifier: at full recall F1 is pinned by prevalence
to 2p/(1+p) = 0.5185 at 35% prevalence. Deriving the Bayes optimum directly
from the data generating process (src/generator_ceiling.py, no model involved)
puts the best achievable F1 under this metric at 0.5226.


================================================================================
EXTERNAL RESOURCES USED  (per competition rules)
================================================================================

All work is original. No pre-trained models are used in the submitted solution.
Open-source libraries only, each under its own license:

  pandas, numpy, scipy, scikit-learn   data handling, CV, metrics, calibration
  lightgbm, xgboost, catboost          gradient boosted tree models
  joblib                               model serialisation
  pyarrow                              parquet feature cache
  matplotlib                           figures

Benchmarked but NOT part of the submitted solution:
  autogluon 1.6.1  - AutoML reference point (src/bench_autogluon.py). Competitive
                     but not shipped: ~0.008 skill difference for a 274 MB
                     artifact.
  tabpfn 8.4.0     - pretrained tabular model; could not be evaluated, its
                     package requires a one-time license click-through.

Dataset: only the organisers' provided competition data. No external data.


================================================================================
CONTACT / NOTES
================================================================================

Everything reported here is out-of-fold on the 3,000 training athletes; the test
labels were never available to us. The full reasoning, including the approaches
that failed, is in docs/findings.md.
================================================================================
