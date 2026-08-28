# What the data actually is

Notes from probing the PlayHack ML Track dataset. These are the results worth
putting in the Round-1 deck, in roughly the order they should be told.

## 1. The split is a trap

| split | days of wearable data | window |
|---|---|---|
| train | 60 | 2026-01-05 … 2026-03-05 |
| test  | 30 | 2026-01-05 … 2026-02-03 |

Days 31–60 of train **are the risk window the labels describe**. Any feature
averaged over "all of train" silently reads the answer. Train CV looks great,
the leaderboard collapses. Every feature is therefore clipped to the first 30
days for both splits, and `features.assert_no_leakage()` re-checks it.

## 2. Injury is a hazard process, not a coin flip

Deciles of `steps_acwr` (acute:chronic workload ratio — last 7 days vs prior 23):

| decile | injury rate | mean onset day |
|---|---|---|
| 1–8 | 0.16 – 0.29 | ~22 |
| 9 | 0.70 | 12.4 |
| 10 | **1.00** | **5.0** |

Two mechanisms are visible:

- **Background hazard** — low ACWR, ~20% injury rate, onset scattered late
  (mean 21.9, sd 5.8). Largely irreducible.
- **Overload hazard** — switches on around ACWR ≈ 1.13 and rises to *certainty*.
  Above ACWR > 1.15 the injury rate is 0.94 and onset is tightly determined
  (correlation with ACWR −0.92, residual sd 1.5 days).

This is why the same features drive both heads: `injured` and `onset` are one
latent time-to-injury T, coarsened as `injured = 1(T ≤ 30)`.

## 3. The targets have very different amounts of signal

| target | best single feature | correlation |
|---|---|---|
| `injured_in_risk_window` | `hr_acwr` | r = 0.55 (AUC 0.758) |
| `onset_day_offset` | `steps_monotony` | **r = 0.87** |
| `recovery_duration` | anything | **r ≤ 0.06** |

`recovery_duration` is not predictable from wearable data at all. It is set by
**sport**: Basketball 14.5 days, Football 14.1, everything else ≈ 10. Within a
sport it is noise — after conditioning on position the largest correlation with
any of the 223 features is 0.067. The right model here is a conditional
*median*, not a fitted curve, and an L2 objective actively hurts: the original
L2 regressor scored MAE 3.005, **worse than just predicting the per-sport
median (2.86)**.

## 4. The ramp proxies are one latent, measured almost perfectly

The `_acwr` family (`hr_acwr`, `steps_acwr`, `cal_acwr`, `intens_acwr`, …)
correlate **r ≈ 0.99 with each other** but only 0.86 with onset.

That is the important asymmetry: the latent is measured nearly without error,
so the gap to the target is *irreducible noise*, not something better features
would recover. Averaging the proxies (PCA1, mean-of-z over 5/10/20/40/80 of
them) does not improve on the best single one — they share the same noise, so
there is nothing to average away.

This is what makes a Bayes ceiling estimable — see `src/oracle.py`.

## 5. Two things that sounded good and lost

Recorded because negative results belong in the deck too.

- **AFT survival model** (`src/survival.py`). Since `injured` and `onset` are
  one censored T, an accelerated-failure-time model should use all 3000
  athletes (1950 right-censored at day 30) instead of fitting the onset head on
  only the 1050 injured. It lost on both heads: AUC 0.7466 vs 0.7508, onset MAE
  2.868 vs 2.694. The lognormal AFT is mis-specified for a two-mechanism
  (background + overload) generator; forcing one parametric survival curve
  costs more than the extra rows gain.

- **More features.** The full 223-column matrix scores *below a single raw
  `hr_acwr` column*. The ACWR family is so redundant that trees spread splits
  across near-duplicates and overfit. Selecting ~35 columns inside each CV fold
  recovers the loss:

  | k | classifier AUC | onset MAE |
  |---|---|---|
  | 3 | 0.7330 | 2.881 |
  | 12 | 0.7463 | 2.683 |
  | **35** | **0.7572** | **2.649** |
  | 100 | 0.7522 | 2.695 |
  | 206 | 0.7497 | 2.703 |

## 6. The fix for `recovery_duration`: blend the group median in, don't fight it

Section 3 said recovery has no per-athlete signal. The consequence isn't "give
up" — it's "stop asking trees to rediscover the group mean from noise." A
cross-fitted sport+gender median (MAE 2.898) beats every tree model on its own
(LightGBM/XGBoost/CatBoost: 2.928–2.948, `src/recovery_v2.py`). Adding it as a
4th candidate to the weighted blend takes 62% of the blend weight and pulls
the ensemble to MAE 2.895 — closing gap-to-ceiling from 76% to 86% (ceiling
2.841, constant baseline 3.233). The trees weren't wrong to include; they just
needed a strong simple baseline to lean on instead of overfitting noise trying
to beat it outright.

## 7. The classifier threshold is bimodal, not miscalibrated

> **Superseded by §8.** Everything below is correct *about F1*, and the bimodality
> it documents is real and still load-bearing. But F1 turned out not to be the
> whole objective: the organisers' metric penalises missed injuries so heavily
> that the right threshold sits far below this section's 0.32. Kept because the
> distributional finding still explains *why* the F1 optimum lands where it does.


Test's calibrated injury probabilities average 0.385 — noticeably above
train's 35% base rate — but the F1-optimal decision threshold (0.32) still
only flags 25.5% of test as positive. That looks like a bug (why doesn't the
positive rate match the mean probability?) until you look at the actual
distribution: it's sharply bimodal, ~78% of test athletes cluster at
p≈0.15–0.30 (background hazard) and ~23% at p≥0.65 (overload hazard), with
almost nothing in the gap between. F1 wants the threshold to sit in that gap
and isolate the overload cluster; matching the mean instead would flag every
background-hazard athlete too, and each only has a ~23% chance of really being
positive — a lot of false positives for very little recall. **Don't
quantile-match a threshold to a target prevalence without checking the shape
first**: for a bimodal score, the mean is the wrong target for an F1 decision.

The 0.385 isn't wasted, though — it cross-validates independently. Test has
22.0% of athletes above the overload ACWR threshold vs 17.5% in train, with
matching conditional injury rates (~0.93 overload / ~0.23 background) in both
splits — a pure covariate shift, not a labelling shift. Reconstructing the
expected test prevalence from that decomposition across four different ACWR
proxies (steps/hr/load/cal) lands at 0.371–0.385, matching the model's
calibrated mean almost exactly. The model learned the right thing; the
question was only ever which number to report where.

## 8. The metric pays for recall, not precision — and we were optimising the wrong thing

We spent most of this project tuning against F1 because we could not find a
published metric. It is in the problem-statement PDF, page 6, behind the
**Brief / Case** button on Unstop (transcribed in `docs/problem_statement.md`,
implemented in `src/score.py`). It changes the answer completely.

Two clauses do all the work:

- Task B is scored over **every test athlete truly injured** — the *actual*
  positives, not our predicted ones. A false positive is simply not in that
  population, so it costs Task B **nothing**; it only dilutes Task A precision.
- A **missed** injury costs a flat `PENALTY = 30` on *both* timing heads.

The baselines are small — the organisers' baseline predicts the training mean for
every injured athlete, giving MAE **7.6148** (onset) and **3.2416** (recovery). So
a single miss is worth ~4 baseline-units of onset error and ~9 of recovery. Setting
`recall·MAE_hit + (1-recall)·30 < MAE_baseline` and solving for our measured hit
errors (2.62 / 2.93) gives the recall needed for *any* nonzero skill:

| head | baseline MAE | break-even recall |
|---|---|---|
| `onset_day_offset` | 7.6148 | **0.818** |
| `recovery_duration` | 3.2416 | **0.988** |

Recovery is knife-edge: it needs essentially perfect recall. Measured on OOF:

| threshold | pos rate | recall | F1 | skill_onset | skill_recovery | mean |
|---|---|---|---|---|---|---|
| 0.40 (**F1-optimal**) | 0.212 | 0.511 | **0.637** | 0.000 | 0.000 | 0.213 |
| 0.30 | 0.296 | 0.575 | 0.623 | 0.000 | 0.000 | 0.208 |
| 0.10 | 0.952 | 0.983 | 0.529 | 0.598 | 0.000 | 0.375 |
| **~0.05 (shipped)** | 0.991 | **1.000** | 0.522 | **0.656** | **0.107** | **0.428** |

**The F1-optimal threshold scores 0.000 on both Task B components.** Trading
0.115 of F1 for 0.76 of combined skill roughly doubles a balanced score.

The selection rule is deliberately *not* the argmax. The sweep's optimum excludes
~0.9% of athletes for +0.004 F1, but if that exclusion clips even a handful of
genuinely injured test athletes, recovery skill collapses from ~0.11 to 0 — upside
0.004 against downside 0.107. So `src/rethreshold.py` takes **half** the maximum
OOF-safe exclusion, and applies it as a **rank quantile** rather than an absolute
probability (OOF rows are scored by fold-models averaged over 3 seeds, test rows by
single full-data fits — the same distribution mismatch that made an earlier
absolute threshold fire on 25% of test when it should have fired on ~38%).

Sensitivity, since the PDF never states how Task A and Task B combine: the low
threshold wins for any Task-A weight up to ~0.78, and only loses if Task A carries
≥80% of the final score. Given the PDF presents them as co-equal sections, low
threshold is the defensible read — but it *is* a judgement call, and worth a line
on the evaluation slide.

**Transferable lesson:** find the metric before optimising. We built a genuinely
good ranking model — the AUC and ceiling analysis in §1–6 all hold — and then
pointed it at the wrong decision rule, which would have thrown away half the
available score at the very last step.

## 9. Verifying the ceiling from the data itself

The §8 ceiling was estimated from our own out-of-fold predictions, which is
circular — a weak model yields a low "ceiling". So `src/generator_ceiling.py`
reverse-engineers the generator from the dataset and derives the Bayes-optimal
score analytically, with no model involved.

**What the data says the generator is:**

- **Injury is deterministic at the top.** `steps_acwr > 1.64` → injury rate
  **1.0000**, n=300, zero exceptions; `> 1.50` → 0.9972. Below ~1.05 the rate is
  a flat ~0.20–0.23 background hazard, with a smooth transition between. So ~12%
  of athletes are *certain* to be injured and the rest carry low random risk —
  the two-mechanism picture in §2, but with a smooth ramp rather than a clean step.
- **Onset is a tight decreasing function of the same ramp**
  (|spearman| = 0.85). In the top ramp bins the conditional spread collapses to
  **~1 day** (mean 3.1, sd 1.6); in the background bins it is diffuse (mean 22.6,
  sd 5.2). A steeper ramp breaks the athlete both *more surely* and *sooner*.
- **Recovery is sport, and nothing else.** Best correlation over all 223 features
  is **0.05** — noise. Contact sports run long (Basketball 14.5, Football 14.1)
  versus ~10 for Athletics/Badminton/Tennis/Volleyball. Cross-fitted, `sport+gender`
  is the best grouping (MAE 2.894); adding `position` *hurts* (3.045) by splitting
  cells too thin.

**What that confirms:**

| | ours | data-derived Bayes | model-conditioned | ceiling used | closed |
|---|---|---|---|---|---|
| F1 | 0.5210 | 0.5226 | 0.5246 | 0.5246 | 99.3% |
| skill_onset | 0.6541 | 0.6285 | 0.6647 | 0.6647 | 98.4% |
| skill_recovery | 0.1075 | 0.1073 | 0.1329 | 0.1329 | 80.9% |
| mean of three | 0.4275 | 0.4195 | 0.4407 | 0.4407 | 97.0% |

1. **F1 ≈ 0.52 is not a weak model — it is pinned by the metric.** At full recall
   F1 collapses to the identity `2·prevalence/(1+prevalence)` = **0.5185** at 35%
   prevalence. Bayes-optimal under this metric is 0.5226; we score 0.5210, i.e.
   **99.7% of what is achievable**. No model can do better without abandoning
   recall, and abandoning recall costs far more than it gains.
2. **Bayes-optimal play flags 99.0% of athletes** — independently reproducing the
   §8 decision from the generator rather than from our own model.
3. **Chasing F1 halves the score**: Bayes mean-of-three is 0.4195 playing the
   metric versus 0.2101 playing F1.

**A ceiling is only as good as its estimator.** Bound a 35-feature model with a
monotone fit on one feature and the model "exceeds the maximum" — which is a
statement about the bound, not the model. Two earlier versions of this analysis
did exactly that (113% and 108.9%) before conditioning on enough. Both ceiling
scripts now warn when a score exceeds them, and the reported ceiling is the
*highest* (tightest) estimate across methods.

## 10. Modelling consequences

1. Clip every feature to the 30-day observation window.
2. Build ACWR / monotony / strain explicitly — they are the generator's driver.
3. Select ~20–35 features per head, inside the fold.
4. Use **L1 objectives** for the day-count heads: MAE wants the conditional
   median.
5. For `recovery_duration`, blend a cross-fitted group median in alongside the
   trees rather than expecting trees to find it unaided.
6. Before "fixing" a threshold/prevalence mismatch, plot the score
   distribution — bimodal structure changes what the right target even is.
7. **Read the metric first.** Under this one, a missed injury costs 30 against a
   3.24 baseline, so recall dominates precision and the threshold belongs near
   the bottom of the score range, not at the F1 optimum.
8. When a metric is knife-edge (recovery needs recall ≈ 0.99), pick the operating
   point with **margin**, not the argmax — and transfer it across splits as a
   rank quantile, since OOF and test probabilities are not on the same scale.
