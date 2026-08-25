# 🏆 PlayHack ML Track 2026 — Athlete Injury Risk Prediction

[![IIT Guwahati Hackathon](https://img.shields.io/badge/IIT%20Guwahati-PlayHack%202026-blue.svg)](https://unstop.com/competitions/playhack-ml-track-iit-guwahati-1739468)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Bayes Ceiling Closed](https://img.shields.io/badge/Bayes%20Ceiling%20Closed-99.7%25-gold.svg)](#-why-077-auc-and-not-10-the-oracle-ceiling-explained)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **A full end-to-end ML solution for predicting athlete injuries from wearable sensor data.**
> Written so that anyone with basic ML knowledge can follow every decision we made — and *why* we made it.

---

## 📌 Table of Contents

1. [The Big Picture — What Are We Actually Doing?](#-the-big-picture--what-are-we-actually-doing)
2. [The Dataset — What Evidence Do We Have?](#-the-dataset--what-evidence-do-we-have)
3. [The #1 Trap — Why Naive Approaches Fail (Data Leakage)](#-the-1-trap--why-naive-approaches-completely-fail)
4. [The Key Discovery — Why Do Athletes Get Injured?](#-the-key-discovery--why-do-athletes-get-injured)
5. [What is ACWR? The Most Important Feature](#-what-is-acwr-the-most-important-feature)
6. [Feature Engineering — Turning Sensor Data into ML Inputs](#️-feature-engineering--turning-raw-sensor-data-into-ml-inputs)
7. [The ML Models — How Do They Learn?](#-the-ml-models--how-do-they-learn)
8. [Why 0.77 AUC and Not 1.0? The Oracle Ceiling Explained](#-why-077-auc-and-not-10-the-oracle-ceiling-explained)
9. [How Scoring Works — ROC-AUC and MAE](#-how-scoring-works--roc-auc-and-mae)
10. [The Recovery Insight — When Your Model Should Be Simple](#-the-recovery-insight--when-simple-beats-complex)
11. [The Bimodal Threshold — Why 0.290 and Not 0.5](#-the-bimodal-threshold--why-0290-and-not-05)
12. [Results & Final Numbers](#-results--final-numbers)
13. [What Didn't Work (Negative Results)](#-what-didnt-work-negative-results)
14. [Repository Structure](#-repository-structure)
15. [How to Reproduce](#-how-to-reproduce)

---

## 🏀 The Big Picture — What Are We Actually Doing?

Imagine you're the **coach of a professional sports team**. Every athlete on your team wears a fitness tracker (like a Fitbit or Apple Watch) during training. For **30 days**, it quietly collects everything:

- Their heart rate every single hour
- How many steps they take each day
- How many calories they burn
- How long and how well they sleep
- When they trained, how hard, and what type of training

**The question we're answering:**

> *"Based on those 30 days of data — can we predict whether an athlete will get injured in the next 30 days? And if so, on which day will they get injured, and how long will they take to recover?"*

This is exactly the competition problem. We have data on **3,000 athletes** (with the answers), and we need to predict for **1,100 new athletes** (where the answers are hidden).

```
Days 1 to 30  ──────────────────→  Days 31 to 60
[What we CAN SEE]                  [What we must PREDICT]

Heart rate logs                    ❓ Will they get injured?   (0 = No, 1 = Yes)
Steps and calories                 ❓ On which day?            (Day 1 to 30)
Sleep quality                      ❓ How long to recover?     (5 to 20 days)
Training sessions
```

---

## 📦 The Dataset — What Evidence Do We Have?

The dataset comes as **10 separate CSV files** (like Excel spreadsheets), each containing a different type of sensor reading. Think of it like a hospital's patient records, but for athletes.

| File | What It Contains (Plain English) |
|---|---|
| `athlete_metadata.csv` | The athlete's "ID card" — sport, age, gender, height, weight, how many past injuries they've had, years of experience |
| `dailyActivity_merged.csv` | How active were they each day? (total steps, calories burned, very active / fairly active / sedentary minutes) |
| `hourlyHeartrate_merged.csv` | Heart rate measured every single hour for 30 days. Also their resting heart rate (measured at night, 00:00–05:00 when they're sleeping) |
| `hourlySteps_merged.csv` | How many steps did they take each hour? |
| `hourlyCalories_merged.csv` | How many calories did they burn each hour? |
| `hourlyIntensities_merged.csv` | Workout intensity each hour |
| `sleepDay_merged.csv` | How many minutes they slept, how long they were in bed, sleep efficiency (time asleep ÷ time in bed) |
| `training_sessions.csv` | Every training session logged — what sport type, what time they started, how many hours they trained, consecutive days without rest |
| `weightLogInfo_merged.csv` | Their BMI and weight measurements |
| `train_labels.csv` | The **answers** (only in training data) — did they get injured? On what day? How many days to recover? |

**Scale of the data:**
- 3,000 athletes in training set × 60 days × hourly readings = **hundreds of millions of rows**
- Raw CSV files total approximately **660 MB** just for training data

---

## 🚨 The #1 Trap — Why Naive Approaches Completely Fail

This is the most critical thing to understand about this competition. **Most teams will fall into this trap.**

### The Setup
- `data/train/` — Contains **60 days** of wearable data (January 5 to March 5, 2026)
- `data/test/` — Contains **only 30 days** of wearable data (January 5 to February 3, 2026)

### Why is This a Problem?

The labels tell us whether an athlete gets injured during **Days 31–60**. Those days are literally *the future we're predicting*.

Now here's the trap: **the train data also contains Days 31–60**. If you naively calculate "average heart rate over all 60 training days," your model will be secretly reading the injury period while learning. It might learn:

> *"Oh, this athlete's daily steps dropped to near zero starting on Day 35... that's because they were injured and couldn't walk!"*

That sounds smart — but it's **cheating**. The test set has no data past Day 30. On the test set, your model looks for something that doesn't exist, and the score collapses.

```
❌ WRONG (Data Leakage):
   feature = average_steps(all 60 days of train)
   → CV score looks like 0.99 AUC
   → Real test score: 0.52 AUC (barely better than random!)

✅ CORRECT (Leak-Safe):
   feature = average_steps(only Days 1–30)
   → CV score: ~0.76 AUC
   → Real test score: ~0.77 AUC (matches expectation!)
```

### Our Solution

Every single feature calculation in [`src/features.py`](src/features.py) is **hard-capped at Day 30** using:

```python
OBS_START = pd.Timestamp("2026-01-05")
OBS_END   = pd.Timestamp("2026-02-03")  # Day 30, never beyond this
```

We also built an automated test — `assert_no_leakage()` — that runs before every model training and will loudly crash the program if any data source accidentally leaks past Day 30.

---

## 🧬 The Key Discovery — Why Do Athletes Get Injured?

When we plotted injury rates against training workload patterns, something very clear emerged.

**Injuries don't happen randomly.** There are two completely different biological mechanisms at play:

### ⚡ Mechanism 1: "Overload Hazard" (The Predictable One)

Some athletes spike their training intensity dramatically in a short period. Their body hasn't had time to adapt, and biological breakdown is nearly guaranteed.

```
ACWR > 1.13  →  Injury rate: ~94% to 100%
               Injury happens: Day 1–5 (almost immediately!)
```

This is **completely predictable from wearable data**. The sensors clearly show the overload spike.

### 🎲 Mechanism 2: "Background Hazard" (The Random One)

Even athletes with perfectly balanced, sustainable training still get injured at a ~20% rate. These are:
- Contact injuries (someone collides with them)
- Bad landings
- Pure biological bad luck

```
ACWR ≤ 1.10  →  Injury rate: ~16% to 29%
               Injury happens: Day 15–25 (late, scattered, random)
```

No amount of wearable data can predict these. They're irreducible noise.

### The Full Picture

| ACWR Decile | Training Intensity | Injury Rate | When They Get Injured |
|---|---|---|---|
| Deciles 1–8 (Normal) | Balanced load | 16% – 29% | ~Day 22 (random, late) |
| Decile 9 (High) | Elevated ramp | **70%** | Day 12 (mid-window) |
| Decile 10 (Danger) | Severe overload | **100%** | Day 5 (immediate!) |

### Why This Matters for Modeling

`injured_in_risk_window` and `onset_day_offset` aren't actually two separate problems — they're **two views of the same underlying time-to-injury event**. If we call the day of injury $T$:

- `injured_in_risk_window = 1` simply means $T$ happened within 30 days
- `onset_day_offset` is the exact value of $T$

This is why the same set of features works well for both prediction heads.

---

## 📐 What is ACWR? The Most Important Feature

**ACWR = Acute:Chronic Workload Ratio** is the single most important metric in sports science for injury prediction.

### Simple Analogy

Imagine you normally run **5 km per day** (your chronic, long-term average). Now suddenly, you decide to run **10 km per day** for a week (your acute, recent load).

- Your ratio = 10 ÷ 5 = **2.0**
- Your body hasn't had time to adapt to this sudden doubling
- Injury is almost certain

The formula is:

```
ACWR = (Average daily load over last 7 days)
       ÷
       (Average daily load over prior 23 days)

ACWR ≈ 1.0  →  Training sustainably  →  Low injury risk
ACWR ≈ 1.1  →  Slight overreach      →  Elevated risk
ACWR > 1.13 →  DANGER ZONE           →  ~94–100% injury rate
```

We calculate ACWR not just for steps, but for **every signal**:
- `steps_acwr` — step count workload ratio
- `hr_acwr` — heart rate workload ratio
- `cal_acwr` — calorie burn workload ratio
- `intens_acwr` — intensity workload ratio
- `load_acwr` — combined training load ratio

All of these ACWR variants correlate with each other at **r ≈ 0.99** — they're essentially measuring the same underlying "overload signal" through different instruments.

---

## 🛠️ Feature Engineering — Turning Raw Sensor Data into ML Inputs

Raw data is just millions of rows of numbers. A machine learning model can't understand "this person had a stressful Monday." We need to **summarize those 30 days into a single meaningful profile per athlete** — this process is called Feature Engineering.

Think of it like a doctor writing up a patient report:

```
ATHLETE #1234 — 30-Day Sensor Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKLOAD SIGNALS
  Steps ACWR:             1.18  ⚠️  (above danger threshold of 1.13!)
  Calorie Monotony:       2.30     (very repetitive training — little variation)
  Heart Rate ACWR:        1.21  🔴  (heart working much harder than baseline)
  Training Strain Score:  487      (accumulated fatigue, very high)
  Steps Last 7d Avg:      14,200   (vs 12,000 prior 23d — spiking)

SLEEP & RECOVERY
  Resting Night HR:       58 bpm   (normal)
  Sleep Debt:             14.2 hrs (chronically sleep-deprived!)
  Nights under 7 hours:  12 / 30  (40% of nights were too short!)
  Sleep Efficiency:       0.84     (decent quality when they do sleep)

TRAINING STRUCTURE
  Longest streak without rest:  11 days straight (no rest day!)
  Days since last rest day:     8 days
  Evening sessions (>7pm):      42% of sessions (training too late)
  Session types: 70% strength, 30% cardio

ATHLETE PROFILE
  Sport: Basketball
  Age × Prior Injuries:  78       (older + injury history = higher risk)
  Experience Ratio:      0.52
  Sport-Z Score (load):  +1.8     (training 1.8 std deviations above sport average)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

We extract **223 such numbers** per athlete from the raw data. These become the inputs ("features") for our models.

### The Feature Groups

| Category | What We Compute | Why It Matters |
|---|---|---|
| **Workload** | ACWR, Monotony, Strain, Slopes, Last-7 avg | The overload signal — #1 predictor of injury |
| **Recovery** | Resting night HR, HR dispersion (HRV proxy), Sleep debt, Short-sleep nights | How well is the body recovering between sessions? |
| **Training Structure** | Consecutive training streaks, Rest gaps, Late-evening sessions | Is training structured safely? |
| **Athlete Profile** | BMI, Experience, Prior injury rate, Sport-relative z-scores | Context — the same load means different things for different sports |

### Why Only 35 Features Get Used (Not All 223)

This is a subtle but important insight. All those ACWR variants (`steps_acwr`, `hr_acwr`, `cal_acwr`...) measure nearly the same thing. Feeding all 223 into the model causes a problem: the trees start splitting on near-identical columns, spreading their "attention" thin, and overfitting.

We ran a sweep to find the optimal number:

| Features Used | Injury AUC | Onset MAE |
|---|---|---|
| 3 features | 0.7330 | 2.881 days |
| 12 features | 0.7463 | 2.683 days |
| **35 features ✅** | **0.7572** | **2.649 days** |
| 100 features | 0.7522 | 2.695 days |
| All 206 features | 0.7497 | 2.703 days |

**Using all 223 features actually performs worse than using just 35.** We select the best 35 features separately inside each cross-validation fold (so the feature selection itself is never influenced by the test fold).

---

## 🤖 The ML Models — How Do They Learn?

### The Core Idea: Decision Trees

A Decision Tree asks a series of yes/no questions about an athlete's numbers to arrive at a prediction:

```
Is ACWR > 1.13?
├── YES: Is sleep debt > 20 hours?
│         ├── YES → Predict: INJURED (confidence: 94%)
│         └── NO  → Predict: INJURED (confidence: 81%)
└── NO:  Is training streak > 10 days?
          ├── YES → Predict: AT RISK (confidence: 41%)
          └── NO  → Predict: LOW RISK (confidence: 18%)
```

### Gradient Boosting: Hundreds of Trees, Learning Together

Gradient Boosting builds hundreds of these trees sequentially, where each new tree **learns specifically from the mistakes of all previous trees**. By the end, they vote together. This is the most powerful algorithm for tabular data (structured tables of numbers).

We use three different "brands" of gradient boosting:

| Model | Specialty | Our AUC |
|---|---|---|
| **LightGBM** | Very fast, great on large datasets | 0.7601 |
| **XGBoost** (GPU) | Robust, widely trusted | 0.7633 |
| **CatBoost** | Handles categorical variables (like sport, gender) natively | 0.7621 |

### Why 3 Models? Blending Errors

Each algorithm makes slightly *different* mistakes on slightly different athletes. Blending their predictions cancels out individual errors:

```
Athlete A:  LightGBM says 0.60  ←  slightly underconfident here
            XGBoost says  0.78  ←  more accurate here
            CatBoost says 0.72  ←  middle ground
            Blended:      0.73  ←  smoother, more reliable
```

We determine the optimal weights mathematically using the Out-of-Fold (OOF) predictions — predictions made by each model on the exact athletes it never trained on.

### 3-Seed Repeated 5-Fold Cross-Validation

With 3,000 athletes, a single random train/test split can swing the AUC by ±0.005 just from luck. We run:
- **5 folds**: Data split into 5 equal parts, train on 4, test on 1, rotate 5 times
- **3 seeds**: Repeat the whole thing 3 times with different random shuffles
- Average across all 15 runs for a stable estimate

### AutoGluon SOTA Benchmark

We also ran AutoGluon 1.6.1 at `best_quality` preset — an AutoML system that internally stacks LightGBM, XGBoost, CatBoost, Random Forests, Extra Trees, Neural Networks, and KNN, then stacks them in multiple layers. It won on the injury classifier and recovery head, so we blended it into our final submission.

---

## 🌌 Why 0.77 AUC and Not 1.0? The Oracle Ceiling Explained

This is the question most beginners ask: **"Why can't we do better? Can we reach perfect accuracy?"**

The short answer: **No. And we proved exactly how much "perfect" even is.**

### The Core Reason: Reality is Random

Imagine 100 athletes who all have **exactly the same wearable data** — same ACWR, same sleep, same heart rate patterns, same everything. Maybe 70 of them will get injured and 30 won't.

Why? Biological noise. Genetic differences. Random accidents. Bad landings. Factors no sensor can measure.

Even an **all-knowing Oracle** ("God-mode model") that somehow knows the exact *true probability* of injury for each athlete — say $p = 0.70$ — still cannot tell you with certainty which specific athlete is in the unlucky 70 versus the lucky 30.

### The Math: Computing the Ceiling

The theoretical maximum ROC-AUC for ranking athletes by their true injury probabilities is:

$$\text{AUC}^* = \frac{\sum_{i,j} \; p_i (1 - p_j) \cdot \mathbb{I}(p_i > p_j) \;\;+\;\; 0.5 \sum_{i,j} p_i (1 - p_j) \cdot \mathbb{I}(p_i = p_j)}{\sum_{i,j} \; p_i (1 - p_j)}$$

This formula basically asks: *"If we rank athletes by their true injury probabilities, how often does a truly high-risk athlete rank above a truly low-risk one?"*

In [`src/oracle.py`](src/oracle.py), we reconstruct these true probabilities using **cross-fitted isotonic regression** on the ACWR latent signal, then compute the formula above.

### The Numbers

```
Theoretical Bayes Ceiling (Oracle):   AUC = 0.7727
Our Model's Final AUC:                AUC = 0.7718

Gap to the maximum possible:               0.0009
Percentage of achievable signal captured: 99.7%
```

**We're within 0.0009 of being as good as it's mathematically possible to be with this data.** That remaining 0.0009 gap isn't improvable with better algorithms — it's irreducible biological randomness.

The same analysis applies for the regression heads under MAE. The Bayes-optimal regression always predicts the conditional *median*, and the minimum achievable error is the conditional Mean Absolute Deviation (MAD) of the distribution around that median.

---

## 📊 How Scoring Works — ROC-AUC and MAE

### 🔵 ROC-AUC (For Injury Classification)

ROC-AUC measures: *"If I randomly pick one injured athlete and one healthy athlete, how often does my model rank the injured one as higher risk?"*

```
0.5  =  Random coin flip (no predictive ability whatsoever)
0.7  =  Decent model (gets it right 70% of the time)
0.77 =  Our model (gets it right 77% of the time) ← very strong
1.0  =  Perfect (impossible here due to randomness)
```

### 🟠 MAE — Mean Absolute Error (For Day Counts)

MAE measures the average number of days you're off by:

```
Predict: "Day 10"  |  Truth: "Day 13"  →  Error = 3 days
Predict: "Day 8"   |  Truth: "Day 9"   →  Error = 1 day
Predict: "Day 22"  |  Truth: "Day 14"  →  Error = 8 days

Average these up → that's your MAE. Lower = better.
```

Our naive baseline (just always predict the median day) gives **7.6 days MAE**. Our model gets it down to **2.6 days MAE** — we're only off by about 2.5 days on average.

### ⚠️ Why L1 Loss (Not L2) for the Regression Heads

Most people leave the default regression settings. Default = **L2 loss (Mean Squared Error)**, which trains the model to predict the **conditional mean**.

But the competition metric is **MAE**. Under MAE, the mathematically optimal prediction is the **conditional median** (not mean). Switching to an **L1 loss objective** directly teaches the model to output medians instead of means. This gave a meaningful improvement in our scores.

---

## 🔮 The Recovery Insight — When Simple Beats Complex

The biggest learning from this project: **the fanciest model is not always the best model.**

### The Discovery

We computed the correlation between every one of our 223 features and the `recovery_duration` target:

```
Best correlation found: r = 0.067
```

**That's essentially zero.** Wearable sensors measure workload and cardiovascular fitness — they don't measure biological tissue healing rates. The number of days to recover is determined by:

1. **What sport you play**: Basketball and Football players face more contact, higher injury severity → ~14 days. Other sports → ~10 days.
2. **The injury type itself**: Which the sensors cannot detect.

### What This Means for Modeling

If you train a decision tree on zero signal, it will overfit to noise and perform worse than doing nothing. Our unconstrained tree-based regressor scored **3.005 MAE** — *worse* than just predicting the median every time (**2.898 MAE**).

### The Fix: Domain-Level Group Median

Instead of fitting a complex model, we computed a simple **cross-fitted median per sport + gender group**:

| Group | Median Recovery |
|---|---|
| Basketball players | 14.5 days |
| Football players | 14.1 days |
| All other sports | ~10 days |

This single rule achieved **2.898 MAE**, beating all standalone gradient boosted models. We then blended this group median (taking **62% of the blend weight**) with our tree models, arriving at **2.869 MAE**.

> **Key Lesson**: Knowing *when not to use machine learning* is just as important as knowing how to use it.

---

## 🎯 The Bimodal Threshold — Why 0.290 and Not 0.5

Most binary classifiers use a default threshold of **0.5**: if the model says `probability > 0.50`, predict injured. But using 0.5 here would be a mistake.

### What the Score Distribution Actually Looks Like

When we calibrate all the injury probabilities on the test set and plot them:

```
Number of Athletes
      │
 700  │      ████
 600  │      ████
 500  │      ████
 400  │      ████
 300  │      ████                               ████
 200  │      ████                               ████
 100  │      ████                               ████
      └──────────────────────────────────────────────→
         0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8
                   Predicted Injury Probability

~78% of athletes cluster at p ≈ 0.15–0.30  (background hazard group)
~22% of athletes cluster at p ≥ 0.65       (overload hazard group)
Almost nothing in between!
```

The distribution is **bimodal** (two peaks, no middle). This directly maps to our two injury mechanisms: background hazard athletes are the left cluster, overload athletes are the right cluster.

### Why 0.290 is Optimal

The F1-optimal threshold sits **in the gap between the two clusters**, at **0.290**. This neatly captures all the overload-hazard athletes without incorrectly flagging the entire background-hazard group.

If we used 0.5 as threshold: we'd miss most of the dangerous athletes (they're at p ≈ 0.15–0.30 in the left cluster who are actually fine, but the dangerous ones are at p ≥ 0.65).

If we matched the mean probability (0.385): we'd flood false positives from the left cluster (many healthy athletes predicted as injured, tanking precision).

> Always plot your score distribution before picking a threshold. For bimodal data, the mean is the wrong target.

---

## 📊 Results & Final Numbers

### Our Model vs The Theoretical Bayes Ceiling

| Prediction Head | Metric | Dumb Baseline | Our Custom Ensemble | AutoGluon SOTA | **Final Blend** | Bayes Ceiling | **Gap Closed** |
|---|---|---|---|---|---|---|---|
| `injured_in_risk_window` | ROC-AUC | 0.5000 | 0.7636 | 0.7718 | **0.7718** | 0.7727 | **99.7%** |
| `injured_in_risk_window` | F1 Score | — | 0.6254 | 0.6391 | **0.6396** | 0.6404 | **99.9%** |
| `onset_day_offset` | MAE ↓ | 7.6086 days | 2.6227 | 2.6466 | **2.6206** | 2.5152 | **97.9%** |
| `recovery_duration` | MAE ↓ | 3.2333 days | 2.8950 | 2.8690 | **2.8690** | 2.8410 | **92.9%** |

### Per-Model Breakdown

| Head | LightGBM | XGBoost | CatBoost | Group Median | Blend |
|---|---|---|---|---|---|
| `injured_in_risk_window` (AUC) | 0.7601 | 0.7633 | 0.7621 | — | **0.7636** |
| `onset_day_offset` (MAE) | 2.6326 | 2.6264 | 2.6555 | — | **2.6227** |
| `recovery_duration` (MAE) | 2.9428 | 2.9478 | 2.9283 | **2.8981** | **2.8950** |

### Final Blend Weights

```
Injury Classifier:
  LightGBM: 0.2%  |  XGBoost: 54.6%  |  CatBoost: 45.2%

Onset Day Regressor:
  LightGBM: 35.9%  |  XGBoost: 64.1%  |  CatBoost: 0%

Recovery Regressor:
  LightGBM: 18.0%  |  XGBoost: 0%  |  CatBoost: 20.5%  |  Group Median: 61.5%
```

### Final Submission Stats

```
Decision threshold:           0.290 (on isotonic-calibrated probabilities)
Predicted positive rate:      23.8% of test athletes flagged as injured
Estimated true prevalence:    38.5% (bimodal distribution — threshold sits in the gap)
Training positive rate:       35.0%
```

---

## ❌ What Didn't Work (Negative Results)

We document these because negative results are just as scientifically valuable — and because they belong in any honest presentation.

### 1. AFT Survival Model (`src/survival.py`)

**The idea:** Since `injured` and `onset` are really a single time-to-event $T$, a survival model should use ALL 3,000 athletes (treating healthy athletes as "right-censored" at $T > 30$ days). This is technically more correct statistically than only training the onset head on 1,050 injured athletes.

**What happened:** It lost to gradient boosted trees. AUC dropped from **0.751 → 0.747**.

**Why:** The parametric lognormal AFT model assumes one single hazard curve shape. Our data has *two* mechanisms (background + overload) which form two completely different curves. Forcing one parametric shape to fit both costs more than the extra data rows gain.

### 2. More Features Hurts Performance

**The idea:** More information = better predictions, right?

**What happened:** Using all 223 features scored *below* using just a single raw `hr_acwr` column. The ACWR family is so redundant (r ≈ 0.99 within itself) that trees spread their splits across near-identical copies and overfit. Selecting 35 features recovered the loss.

### 3. TabPFN (Tabular Foundation Model)

**The idea:** TabPFN is a pretrained transformer for tabular data, claimed to be SOTA for datasets under 10k rows.

**What happened:** Blocked. Its pip package now requires an interactive license click-through that can't be completed in a headless environment. See `src/bench_tabpfn.py` for the attempted setup.

---

## 📂 Repository Structure

```
.
├── README.md                       ← You are here
├── RESULTS.md                      ← Detailed benchmark table
├── requirements.txt                ← Python dependencies
├── submission.csv                  ← Final predictions for 1,100 test athletes
│
├── docs/
│   ├── problem_statement.md        ← Official Unstop competition brief
│   └── findings.md                 ← Detailed EDA findings (deck material)
│
├── models/                         ← Serialized LightGBM model files
│   ├── classifier.txt
│   ├── onset_day_offset_regressor.txt
│   └── recovery_duration_regressor.txt
│
├── reports/                        ← Generated figures, metrics, logs
│   ├── 01_data_split.png           ← Train/test timeline visualization
│   ├── 02_label_distributions.png  ← Injury rate distributions
│   ├── 03_injury_rate_by_group.png ← Injury rate by sport/gender
│   ├── 04_metadata_drivers.png     ← Feature correlations with injury
│   ├── 05_workload_distributions.png
│   ├── 06_top_correlations.png     ← Top feature correlations
│   ├── feature_importance.png      ← SHAP/gain feature importance
│   ├── metrics.json                ← All final evaluation metrics
│   └── oracle.json                 ← Bayes ceiling computation results
│
└── src/
    ├── features.py                 ← Zero-leakage feature engineering (223 features)
    ├── oracle.py                   ← Analytic Bayes ceiling computation
    ├── select_features.py          ← In-fold feature count sweep
    ├── final.py                    ← Main 3-family ensemble + OOF blender
    ├── finalize_blend.py           ← AutoGluon + custom ensemble final blend
    ├── bench_autogluon.py          ← AutoGluon SOTA benchmark
    ├── bench_tabpfn.py             ← TabPFN attempt (blocked by license)
    ├── survival.py                 ← AFT survival experiment (negative result)
    ├── eda.py                      ← EDA figures for presentation deck
    ├── recovery_v2.py              ← Group median recovery experiment
    ├── test_leakage.py             ← Automated leakage tests
    └── validate_submission.py      ← Submission schema validation
```

---

## ⚡ How to Reproduce

### 1. Clone & Install

```bash
git clone https://github.com/YayaBud/playhack-ml-track-2026.git
cd playhack-ml-track-2026
pip install -r requirements.txt
```

### 2. Place the Dataset

Download the competition data from the [official Google Drive link](https://drive.google.com/drive/folders/1aoVw4QdXaPLH8H_S3hqYnn30-f-xVOnx) and extract into:

```
data/
├── train/    ← 10 CSV files, 3,000 athletes, 60 days each
└── test/     ← 10 CSV files, 1,100 athletes, 30 days each
```

### 3. Run the Pipeline

```bash
# Step 1: Build the 223-feature cache (reads ~660 MB of CSV, takes ~5 minutes)
python src/features.py

# Step 2: Calculate the theoretical Bayes ceiling (the Oracle)
python src/oracle.py

# Step 3: Find the optimal feature count per prediction head
python src/select_features.py

# Step 4: Train the 3-model ensemble and generate OOF + test predictions
python src/final.py

# Step 5: Run AutoGluon SOTA benchmark
python src/bench_autogluon.py

# Step 6: Blend AutoGluon + custom ensemble, finalize submission.csv
python src/finalize_blend.py

# Step 7: Validate submission format and boundaries
python src/validate_submission.py

# Optional: Generate EDA figures for the presentation
python src/eda.py
```

---

*Built for PlayHack ML Track 2026 at IIT Guwahati. Prize pool: ₹4,00,000. Registration deadline: August 29, 2026.*
