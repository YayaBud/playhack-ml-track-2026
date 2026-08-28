# PlayHack ML Track 2026 — how to run this submission

Athlete injury-risk prediction: given a 30-day observation window of wearable and
training data, predict whether an injury occurs in the following 30-day risk
window, and if so when it starts and how long it sidelines the athlete.

## Install

```bash
pip install -r requirements.txt
```

Python 3.12. The exact library versions the models were pickled under are
recorded in `models/manifest.json` (`versions`).

## Regenerate `submission.csv` from the trained models

```bash
python predict.py --data-root /path/to/data --split test
```

`--data-root` is the folder containing a `test/` subfolder with the ten
competition CSVs (`athlete_metadata.csv`, `dailyActivity_merged.csv`,
`hourly{Steps,Calories,Intensities,Heartrate}_merged.csv`, `sleepDay_merged.csv`,
`weightLogInfo_merged.csv`, `training_sessions.csv`, `sample_submission.csv`).
Defaults to `./data`. Nothing is retrained — this loads `models/*.joblib`,
rebuilds features, blends, thresholds, and writes the CSV.

## Retrain end to end

```bash
python src/features.py            # build the 223 leak-safe features (needs data/train too)
python src/final.py               # repeated-CV ensemble, writes OOF + test arrays
python src/recovery_v2.py         # adds the group-median candidate to recovery
python src/build_final_models.py  # refit on all data, persist models/ + manifest
python src/rethreshold.py         # pick the threshold against the official metric
python src/validate_submission.py # check the submission contract
```

## Checks

```bash
python src/score.py           # self-check of the official metric implementation
python src/test_leakage.py    # asserts no feature can see the risk window
```

## What's in here

| Path | |
|---|---|
| `predict.py` | inference entry point |
| `models/` | persisted models + `manifest.json` (features, blend weights, threshold, versions) |
| `src/score.py` | the organisers' metric (PDF p.6), implemented verbatim |
| `src/features.py` | feature engineering, with the leakage guard |
| `src/rethreshold.py` | threshold selection against the official metric |
| `RESULTS.md` | scores, per-model breakdown, SOTA benchmark |
| `docs/findings.md` | what we learned, including the negative results |
| `docs/problem_statement.md` | the brief and the metric, transcribed |

## The one thing worth knowing

The metric penalises a **missed** injury by a flat 30 on both timing heads,
against baselines of only ~7.6 (onset) and ~3.2 (recovery). A false positive, by
contrast, is not in Task B's evaluation population at all and costs nothing there.
So recall on injured athletes is worth far more than precision: break-even recall
is 0.82 for onset and 0.99 for recovery. Tuning the threshold for F1 alone (the
obvious default) scores **0.000** on both skill components. We therefore flag
almost every athlete and exclude only the most confidently healthy ~1%. Full
derivation in `src/rethreshold.py` and `docs/findings.md`.
