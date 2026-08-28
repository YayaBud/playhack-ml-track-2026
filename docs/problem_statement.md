# PlayHack - ML Track 2026 (IIT Guwahati) — Competition Brief

Source: https://unstop.com/competitions/playhack-ml-track-iit-guwahati-1739468
Dataset: https://drive.google.com/drive/folders/1aoVw4QdXaPLH8H_S3hqYnn30-f-xVOnx

## Round 1 task (verbatim from Unstop "Stages & Timeline")

> Participants will be provided with a sports dataset and specific prediction
> requirements. Teams are required to perform EDA, data preprocessing, feature
> engineering, and predictive modeling to generate predictions for the given
> target(s).
>
> For submission, teams must provide:
> - A PPT presentation covering their EDA, key insights, methodology, models
>   used, evaluation, and results.
> - A ZIP file containing their trained model(s) and the necessary
>   files/code required to run or evaluate the models.
>
> The submission should clearly demonstrate the approach taken, insights
> derived from the dataset, and the performance of the proposed models.

## Round 2 (offline finals, IIT Guwahati campus)

Shortlisted teams build a functional prototype and present/demo live to
judges. Evaluated on implementation, functionality, innovation, feasibility,
overall impact.

## The official metric (problem-statement PDF, page 6)

Reached via the **Brief / Case** button on the Unstop page:
https://d8it4huxumps7.cloudfront.net/uploads/submissions_case/6a8b41d2be715_playhack_ml_ps.pdf

> **Task A: F1 score** — "F1-score, in [0, 1]. Measures whether an injury onset
> occurs during the risk window."
>
> **Task B: timing** — "Evaluated on every test athlete truly injured in the risk
> window. Score `onset_day_offset` and `recovery_duration`."
>
> **Hit: injured** — "Mean absolute error (MAE) is calculated against the true
> `onset_day_offset` and `recovery_duration`."
>
> **Miss: not injured** — "A fixed penalty of PENALTY = n_risk = 30 applies to
> both timing predictions when an injury is missed."
>
> **Skill score** — `skill = max(0, 1 - MAE_model / MAE_baseline)`
>
> **Baseline** — "Predicts the training-set mean onset day and recovery duration
> for every injured athlete."

The PDF does **not** state how Task A and Task B combine into a final ranking.
`src/score.py` implements the above verbatim; `docs/findings.md` §8 works through
what the penalty asymmetry implies for the decision threshold.

Also from the PDF (page 3), a hard submission rule:

> "`onset_day_offset` and `recovery_duration` are required for every athlete in
> `sample_submission.csv`, even when `injured_in_risk_window` is predicted as 0."

*Correction:* an earlier revision of this file claimed no metric was published
anywhere. That was wrong — it was checking the Unstop page body, AMP mirror and
public JSON APIs, none of which expose the PDF behind the Brief/Case button. Every
threshold in this repo before that discovery was tuned for F1 alone, which scores
0.000 on both Task B components.

## Design insight: train=60 days, test=30 days -> risk window is days 31-60

`data/train/*` covers 2026-01-05 to 2026-03-05 (60 days/athlete) for every
table (daily activity, sleep, hourly HR/steps/calories/intensity, sessions).
`data/test/*` covers only 2026-01-05 to 2026-02-03 (30 days). `onset_day_offset`
in train_labels.csv ranges 1-30. This means:
- Days 1-30 = the **observation window** (features come from here in both splits).
- Days 31-60, present ONLY in train, is the **risk window** being predicted —
  `onset_day_offset` counts days into it. Test athletes don't have this data
  because it's literally the future being forecast.
- **Leakage trap**: do not build features from train days 31-60. Any model
  must only see day 1-30 features, mirroring exactly what test provides.

## Targets (inferred from train_labels.csv / sample_submission.csv)

1. `injured_in_risk_window` — binary (0/1)
2. `onset_day_offset` — int, day within the risk window the injury occurs (only defined when target 1 = 1)
3. `recovery_duration` — int, days to recover (only defined when target 1 = 1)

## Registration deadline
29 Aug 2026, 11:59 PM IST (4 days left as of page snapshot 24 Aug 2026)

## Team / eligibility
1-4 members, any Indian college student, inter-college/inter-branch allowed,
one entry per participant, one track OR both allowed.

## Prizes
₹4L pool: Winner ₹1L, 1st RU ₹60k, 2nd RU ₹40k, all + certificate.

## Rules (key)
- Original work only, plagiarism = DQ
- Public APIs/libraries/frameworks/datasets/OSS tools allowed, must disclose
  major external resources/pretrained models used
- Judging panel decision final
