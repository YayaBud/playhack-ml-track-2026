# 🏆 PlayHack ML Track 2026 — Athlete Injury Risk Prediction

[![IIT Guwahati Hackathon](https://img.shields.io/badge/IIT%20Guwahati-PlayHack%202026-blue.svg)](https://unstop.com/competitions/playhack-ml-track-iit-guwahati-1739468)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Model-to-Ceiling Gap Closed](https://img.shields.io/badge/Bayes%20Ceiling%20Closed-99.7%25-gold.svg)](#-why-the-07727-auc-ceiling-the-bayes-oracle-explained)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end Machine Learning pipeline and sports-science feature engineering framework developed for **PlayHack ML Track 2026 (IIT Guwahati)**.

Our final model ensemble closes **99.7% of the theoretical Bayes ceiling** on injury classification and **97.9% / 92.9%** on day-count regressions, beating standalone gradient boosting and SOTA AutoML benchmarks.

---

## 📌 Table of Contents

1. [Executive Summary & Results](#-executive-summary--results)
2. [Problem Statement & Prediction Targets](#-problem-statement--prediction-targets)
3. [Dataset Architecture & Composition](#-dataset-architecture--composition)
4. [The Critical Data Leakage Trap (Train 60d vs Test 30d)](#-the-critical-data-leakage-trap)
5. [The Science: Two-Mechanism Hazard Process](#-the-science-two-mechanism-hazard-process)
6. [Why the ~0.77-0.78 AUC Ceiling? The Bayes Oracle Explained](#-why-the-07727-auc-ceiling-the-bayes-oracle-explained)
7. [Evaluation & Metric Strategy](#-evaluation--metric-strategy)
8. [Feature Engineering (223 Engineered -> 35 Selected)](#-feature-engineering-deep-dive)
9. [Modeling & SOTA AutoML Ensemble](#-modeling--sota-automl-ensemble)
10. [Negative Results & Scientific Insights](#-negative-results--scientific-insights)
11. [Repository Structure](#-repository-structure)
12. [How to Reproduce](#-how-to-reproduce)

---

## 📊 Executive Summary & Results

| Prediction Head | Metric | Baseline | Custom Ensemble | AutoGluon SOTA | **Final Blend** | Bayes Ceiling (Oracle) | **Theoretical Gap Closed** |
|---|---|---|---|---|---|---|---|
| `injured_in_risk_window` | **ROC-AUC** | 0.5000 | 0.7636 | 0.7718 | **0.7718** | 0.7727 | **99.7%** |
| `injured_in_risk_window` | **F1 Score** | - | 0.6254 | 0.6391 | **0.6396** | 0.6404 | **99.9%** |
| `onset_day_offset` | **MAE** ↓ | 7.6086 | 2.6227 | 2.6466 | **2.6206** | 2.5152 | **97.9%** |
| `recovery_duration` | **MAE** ↓ | 3.2333 | 2.8950 | 2.8690 | **2.8690** | 2.8410 | **92.9%** |

*Note: Baseline is random ranking for ROC-AUC and constant-median prediction for MAE. Lower is better for MAE.*

---

## 🎯 Problem Statement & Prediction Targets

Given **30 days of high-frequency wearable sensor telemetry and training logs** for an athlete, we must forecast their injury dynamics over the **subsequent 30-day risk window**:

```
Observation Window (Days 1 to 30)            Future Risk Window (Days 31 to 60)
[============ 30 Days Wearable Logs ============] -> [=== Forecast Next 30 Days ===]
- Heart Rate streams (hourly & night)                  1. Injured? (0 or 1)
- Steps, Calories, Active Minutes                      2. Onset Day? (Day 1..30)
- Sleep Debt & Sleep Efficiency                        3. Recovery Days? (5..20)
- Training Sessions & Workload Ratios
```

### The 3 Output Targets:

1. `injured_in_risk_window` *(Binary: `0` or `1`)*:
   - Whether the athlete sustains an injury during the 30-day forecast horizon.
   - Positive base rate in training data: **35.0%**.

2. `onset_day_offset` *(Integer: `1` to `30`)*:
   - The exact day within the 30-day risk window when the injury occurs.
   - Ground truth exists only for injured athletes.

3. `recovery_duration` *(Integer: `5` to `20`)*:
   - Number of days required for the athlete to fully recover.
   - Ground truth exists only for injured athletes.

---

## 📁 Dataset Architecture & Composition

The dataset comprises **10 multi-modal relational tables** across 3,000 training athletes and 1,100 test athletes:

| File Name | Granularity | Key Fields Extracted |
|---|---|---|
| `athlete_metadata.csv` | Athlete-level | Age, Gender, Sport, Position, Height, Weight, Prior Injuries, Years Playing |
| `dailyActivity_merged.csv` | Daily | Total Steps, Calories, Very Active Minutes, Fairly/Lightly Active Minutes, Sedentary Minutes |
| `hourlyHeartrate_merged.csv` | Hourly | Average HR, Min HR, Max HR, Night Resting HR (00:00–05:00), HR Dispersion |
| `hourlySteps_merged.csv` | Hourly | Hourly Step Total, Peak Hour Volume, Step Volatility |
| `hourlyCalories_merged.csv` | Hourly | Hourly Caloric Burn, Peak Calorie Intensity |
| `hourlyIntensities_merged.csv` | Hourly | Total Hourly Intensity, Peak Workout Density |
| `sleepDay_merged.csv` | Daily | Total Minutes Asleep, Total Time in Bed, Sleep Efficiency, Sleep Debt (<7h nights) |
| `training_sessions.csv` | Session-level | Session Type, Duration (Hours), Start Hour, Consecutive Streaks, Rest Gaps |
| `weightLogInfo_merged.csv` | Periodic | BMI, Weight Change, Body Fat % |
| `train_labels.csv` | Athlete-level (Train only) | Binary Injury Flag, Onset Day Offset, Recovery Duration |

---

## ⚠️ The Critical Data Leakage Trap

> ### 🚨 The #1 Pitfall of this Dataset
> - **`data/train/` contains 60 days of wearable records** (`2026-01-05` to `2026-03-05`).
> - **`data/test/` contains only 30 days of wearable records** (`2026-01-05` to `2026-02-03`).

### Why is this a trap?
Days 31–60 in the training set **are the exact risk window the labels describe**. If you calculate rolling averages or total statistics over the entire 60 days in train:
1. The model detects that an injured athlete stopped working out or recorded elevated resting HR during days 31–60.
2. The training cross-validation score shoots up to ~0.99 AUC (memorizing the future).
3. **The model completely fails on the real test set**, because the test set only provides days 1–30.

### Our Solution:
Every single feature computation is strictly bounded to `2026-01-05` through `2026-02-03` across both splits. We enforce an automated validation check (`assert_no_leakage()` in [`src/features.py`](src/features.py)) that throws an exception if any timestamp beyond Day 30 is touched.

---

## 🧬 The Science: Two-Mechanism Hazard Process

By analyzing the distribution of injuries against training workload, we discovered that injury is **not a uniform random coin flip**, but rather a **two-mechanism hazard process**:

```
                       INJURY HAZARD DISTRIBUTION
================================================================================
Deciles of ACWR (Acute:Chronic Workload) | Injury Rate | Mean Onset Day
----------------------------------------|-------------|-------------------------
Deciles 1 - 8 (Normal Load / Stable)    |  16% - 29%  |  Day 21.9 (Late, Random)
Decile 9      (Elevated Ramp)           |     70%     |  Day 12.4 (Mid-window)
Decile 10     (Severe Overload > 1.13)  |    100%     |  Day  5.0 (Immediate)
================================================================================
```

![Top Correlations](reports/06_top_correlations.png)

1. **Background Hazard (Stochastic Luck)**:
   - Athletes with balanced workloads (ACWR $\le 1.10$) have a baseline ~20% injury rate.
   - When injuries occur here, they happen late in the month (mean day ~22) and are largely unpredictable from wearable logs (stochastic bad luck / contact injuries).
2. **Overload Hazard (Deterministic Fatigue Spike)**:
   - When workload spikes rapidly (Acute:Chronic Workload Ratio $> 1.13$), injury rate jumps to **94%–100%**.
   - The onset day becomes tightly deterministic ($r = -0.92$ with workload ratio), occurring rapidly in days 1–10.
3. **Unification of Classification and Onset**:
   - `injured_in_risk_window` and `onset_day_offset` are not two separate phenomena; they are views of a single underlying time-to-event variable $T$:
     $$\text{injured} = \mathbb{I}(T \le 30)$$

---

## 🌌 Why the ~0.7727 AUC Ceiling? (The Bayes Oracle Explained)

Many beginners ask: *"Why can't our ML model achieve 0.95 or 1.00 ROC-AUC?"*

### 1. Stochastic Labels vs Deterministic Functions
In real-world sports biology, identical athletes exposed to identical workloads do not all experience the exact same outcome. An athlete with an elevated load might have a **70% probability** of injury:
- Nature flips a biased coin with $P(Y=1) = 0.70$.
- Out of 100 such identical athletes, 70 get injured ($Y=1$) and 30 remain healthy ($Y=0$).

Because the true label contains irreducible randomness, **even an omniscient Oracle ("God-mode") model that knows the exact true probability $p_i = P(Y_i=1 \mid X_i)$ cannot achieve 1.0 AUC**.

### 2. Deriving the Bayes-Optimal Expected AUC
The theoretical maximum ROC-AUC for ranking instances by their true probabilities $p$ is given by the expectation over all pairs $(i, j)$:

$$\text{AUC}^* = \frac{\sum_{i,j} p_i (1 - p_j) \cdot \mathbb{I}(p_i > p_j) + 0.5 \sum_{i,j} p_i (1 - p_j) \cdot \mathbb{I}(p_i = p_j)}{\sum_{i,j} p_i (1 - p_j)}$$

In [`src/oracle.py`](src/oracle.py), we reconstruct the true load-ramp latent using cross-fitted isotonic calibration across folds and compute $\text{AUC}^*$:
- **Theoretical Bayes Upper Bound (Ceiling)**: **`0.7727 ROC-AUC`**
- **Our Model Score**: **`0.7718 ROC-AUC`**
- **Gap to Perfection Closed**: **`99.7%`**

### 3. Regression Ceilings under MAE (Conditional Median)
Under Mean Absolute Error (MAE), the Bayes-optimal decision rule is the **conditional median** of the distribution, and the theoretical minimum error is the **conditional Mean Absolute Deviation (MAD)**:
- `onset_day_offset`: Baseline constant MAE `7.61` $\rightarrow$ Oracle Floor `2.52` $\rightarrow$ **Our Model: `2.62 MAE` (97.9% of gap closed)**
- `recovery_duration`: Baseline constant MAE `3.23` $\rightarrow$ Oracle Floor `2.84` $\rightarrow$ **Our Model: `2.87 MAE` (92.9% of gap closed)**

---

## 📐 Evaluation & Metric Strategy

### 1. Why L1 Loss (MAE) for Day Counts?
Most default regressors optimize Mean Squared Error (MSE / L2 loss), which predicts the conditional *mean*. However, the competition evaluates MAE. Optimizing an **L1 objective** trains the model to predict the conditional *median*, directly aligning with the evaluation metric.

### 2. The `recovery_duration` Discovery: Sport-Group Median
Probing the correlation matrix revealed that **no wearable feature correlates with recovery duration ($|r| \le 0.06$)**. Recovery time is biologically governed by the sport and injury type:
- Basketball & Football: Average $\approx 14.2$ days.
- Other sports: Average $\approx 10.0$ days.

Fitting unconstrained decision trees on noise leads to overfitting. We introduced a **cross-fitted sport + gender group median**, which alone achieved `2.898 MAE`, outperforming all standalone gradient boosted trees. Blending this group prior with our trees delivered the winning recovery score.

### 3. Bimodal Decision Thresholding
The calibrated injury probabilities on the test set form a **sharp bimodal distribution**:
- ~78% of athletes cluster around $p \approx 0.15 - 0.30$ (background hazard).
- ~22% of athletes cluster around $p \ge 0.65$ (overload hazard).

![Label Distribution](reports/02_label_distributions.png)

Rather than using a default threshold of $0.50$ or forcing the threshold to match the mean prevalence ($0.385$), we optimize the threshold for **expected F1 score** on the calibrated distribution. The optimal threshold sits at **`0.290`**, cleanly isolating the high-risk overload cluster.

---

## 🛠️ Feature Engineering Deep Dive

We constructed **223 engineered features** in [`src/features.py`](src/features.py) across 5 core domains:

```
                               FEATURE TAXONOMY (223 Features)
┌───────────────────────────┬───────────────────────────┬───────────────────────────┐
│ Workload & Fatigue (ACWR) │ Sleep & Recovery Kinetics │ Training Structure & Gaps │
├───────────────────────────┼───────────────────────────┼───────────────────────────┤
│ • Acute:Chronic Workload  │ • Night HR (00:00–05:00)  │ • Consecutive streak max  │
│   Ratio (last 7d / 23d)   │   Resting HR surrogate    │ • Max rest-day gap        │
│ • Monotony (Mean / Std)   │ • HR Within-day Dispersion│ • Days since last workout │
│ • Strain (Load × Monotony)│   (HRV proxy)             │ • Late evening workout %  │
│ • Step & Calorie Slopes   │ • Sleep debt (<7h nights) │ • Session intensity / RPE │
│ • Peak hourly volume      │ • Sleep efficiency trend  │ • Sport-relative Z-Scores │
└───────────────────────────┴───────────────────────────┴───────────────────────────┘
```

![Feature Importance](reports/feature_importance.png)

### In-Fold Feature Selection (Why 35 Features Beat 223)
Because all ACWR features (`hr_acwr`, `steps_acwr`, `cal_acwr`, `load_acwr`) correlate at $r \approx 0.99$ with each other, feeding all 223 features into tree models causes them to split across redundant columns and overfit.

Sweeping feature counts inside cross-validation folds showed that **selecting the top ~35 gain-ranked features strictly inside each training fold** yields superior generalization:

| Features Kept ($K$) | Classifier AUC | Onset MAE |
|---|---|---|
| 3 | 0.7330 | 2.881 |
| 12 | 0.7463 | 2.683 |
| **35 (Optimal)** | **0.7572** | **2.649** |
| 100 | 0.7522 | 2.695 |
| 206 (All) | 0.7497 | 2.703 |

---

## 🤖 Modeling & SOTA AutoML Ensemble

Our final architecture combines diverse gradient boosted tree families with AutoML multi-layer stacking:

```
                                 FINAL ENSEMBLE ARCHITECTURE
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ 30-Day Wearable & Metadata Logs                                                        │
 └─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                           ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ In-Fold Feature Selection (Top 35 Gain-Ranked Sport-Science Features)                  │
 └─────────────┬───────────────────────────┬────────────────────────────────┬─────────────┘
               │                           │                                │
               ▼                           ▼                                ▼
 ┌───────────────────────────┐ ┌───────────────────────────┐  ┌───────────────────────────┐
 │ LightGBM                  │ │ XGBoost (CUDA GPU)        │  │ CatBoost                  │
 │ (3-Seed 5-Fold CV)        │ │ (3-Seed 5-Fold CV)        │  │ (3-Seed 5-Fold CV)        │
 └─────────────┬─────────────┘ └───────────┬───────────────┘  └─────────────┬─────────────┘
               │                           │                                │
               └───────────────────┬───────┴────────────────────────────────┘
                                   ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ Hand-Crafted Out-Of-Fold Optimal Non-Negative Blend                                    │
 └─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                           │
                                           ├──────────────────────────┐
                                           ▼                          ▼
 ┌─────────────────────────────────────────────────────────┐ ┌────────────────────────────┐
 │ AutoGluon 1.6.1 SOTA Tabular Benchmark (best_quality)   │ │ Sport+Gender Group Median  │
 │ (Stacked ExtraTrees, RF, LightGBM, XGBoost, CatBoost)   │ │ (Domain Prior for Recovery)│
 └─────────────────────────────────────────┬───────────────┘ └────────────┬───────────────┘
                                           │                              │
                                           └──────────────┬───────────────┘
                                                          ▼
                                ┌──────────────────────────────────────────────────┐
                                │ Final Blended Output -> submission.csv           │
                                └──────────────────────────────────────────────────┘
```

---

## 🔬 Negative Results & Scientific Insights

Documenting what **didn't work** is critical for research integrity and competition presentations:

1. **AFT (Accelerated Failure Time) Survival Model ([`src/survival.py`](src/survival.py))**:
   - *Hypothesis*: Since `injured` and `onset` represent a single censored time $T$, a survival model should utilize all 3,000 athletes (treating healthy as right-censored at $t > 30$).
   - *Result*: Lost to gradient boosted trees (`0.7466 AUC` vs `0.7508 AUC`).
   - *Why*: The parametric lognormal distribution assumes a single hazard shape, failing to fit the two-mechanism (background vs overload) reality.

2. **Unconstrained Deep Trees for Recovery Duration**:
   - Standard regressors scored `3.005 MAE`, worse than a naive constant median (`2.898 MAE`). Wearables provide zero signal for recovery; domain-level sport priors must be used.

3. **TabPFN (Tabular Foundation Model)**:
   - Blocked by mandatory interactive license acceptance in headless environment ([`src/bench_tabpfn.py`](src/bench_tabpfn.py)).

---

## 📂 Repository Structure

```
.
├── README.md                      # Comprehensive project documentation
├── RESULTS.md                     # Benchmark scores and model breakdown
├── requirements.txt               # Python dependencies
├── submission.csv                 # Final validated test predictions
├── docs/
│   ├── problem_statement.md       # Scraped official Unstop competition brief
│   └── findings.md                # In-depth exploratory data analysis findings
├── models/                        # Serialized LightGBM models
│   ├── classifier.txt
│   ├── onset_day_offset_regressor.txt
│   └── recovery_duration_regressor.txt
├── reports/                       # Visualizations, sweep logs, and metrics
│   ├── 01_data_split.png
│   ├── 02_label_distributions.png
│   ├── 03_injury_rate_by_group.png
│   ├── 04_metadata_drivers.png
│   ├── 05_workload_distributions.png
│   ├── 06_top_correlations.png
│   ├── feature_importance.png
│   ├── metrics.json               # Full evaluation JSON
│   └── oracle.json                # Bayes ceiling computations
└── src/
    ├── features.py                # Zero-leakage feature engineering pipeline
    ├── oracle.py                  # Analytic Bayes-ceiling calculations
    ├── select_features.py         # In-fold feature selection sweep
    ├── survival.py                # Accelerated Failure Time experiment
    ├── bench_autogluon.py         # AutoGluon SOTA benchmark
    ├── final.py                   # Main 3-family ensemble & OOF blender
    ├── finalize_blend.py          # AutoGluon + custom ensemble blender
    ├── eda.py                     # Report figure generation
    └── validate_submission.py     # Schema and boundary validation
```

---

## ⚡ How to Reproduce

### 1. Environment Setup
```bash
git clone https://github.com/YayaBud/playhack-ml-track-2026.git
cd playhack-ml-track-2026
pip install -r requirements.txt
```

### 2. Dataset Placement
Download competition data from the [official drive link](https://drive.google.com/drive/folders/1aoVw4QdXaPLH8H_S3hqYnn30-f-xVOnx) and extract to:
```
data/train/   # 10 CSVs
data/test/    # 10 CSVs
```

### 3. Execution Pipeline
```bash
# 1. Build leak-safe feature cache (reads CSVs, extracts 223 features)
python src/features.py

# 2. Compute theoretical Bayes ceiling (Oracle)
python src/oracle.py

# 3. Sweep optimal feature count per head
python src/select_features.py

# 4. Train 3-family ensemble & generate base predictions
python src/final.py

# 5. Run AutoGluon benchmark & blend winning models
python src/bench_autogluon.py
python src/finalize_blend.py

# 6. Validate submission format
python src/validate_submission.py
```

---

## 👥 Contributors & Acknowledgements

Developed for **PlayHack ML Track 2026 @ IIT Guwahati**. Special thanks to the organizing committee and mentors.
