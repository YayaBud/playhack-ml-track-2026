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

## 8. Modelling consequences

1. Clip every feature to the 30-day observation window.
2. Build ACWR / monotony / strain explicitly — they are the generator's driver.
3. Select ~20–35 features per head, inside the fold.
4. Use **L1 objectives** for the day-count heads: MAE wants the conditional
   median.
5. For `recovery_duration`, blend a cross-fitted group median in alongside the
   trees rather than expecting trees to find it unaided.
6. Before "fixing" a threshold/prevalence mismatch, plot the score
   distribution — bimodal structure changes what the right target even is.
