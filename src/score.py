"""
The organisers' official Round-1 metric, implemented verbatim.

Source: PlayHack ML Track problem statement PDF, page 6 ("Evaluation metrics"),
linked from the Brief/Case button at
https://unstop.com/competitions/playhack-ml-track-iit-guwahati-1739468
PDF: https://d8it4huxumps7.cloudfront.net/uploads/submissions_case/6a8b41d2be715_playhack_ml_ps.pdf

    Task A: F1 score
        "F1-score, in [0, 1]. Measures whether an injury onset occurs during
        the risk window."

    Task B: timing
        "Evaluated on every test athlete truly injured in the risk window.
        Score onset_day_offset and recovery_duration."

    Hit: injured
        "Mean absolute error (MAE) is calculated against the true
        onset_day_offset and recovery_duration."

    Miss: not injured
        "A fixed penalty of PENALTY = n_risk = 30 applies to both timing
        predictions when an injury is missed."

    Skill score
        skill = max(0, 1 - MAE_model / MAE_baseline)

    Baseline
        "Predicts the training-set mean onset day and recovery duration for
        every injured athlete."

Two consequences drive every modelling decision downstream:

1. Task B's population is the TRULY injured athletes, not the predicted ones.
   A false positive costs Task B nothing (it is simply not in the evaluation
   set); it only dilutes Task A precision. A false negative on a truly injured
   athlete costs a flat 30 on BOTH timing heads.

2. The baselines are small -- onset ~7.61, recovery ~3.24 -- so a penalty of 30
   is worth roughly 4 and 9 baseline-units respectively. Solving
   `recall*MAE_hit + (1-recall)*PENALTY < MAE_baseline` shows a nonzero skill
   needs recall > ~0.82 (onset) and > ~0.99 (recovery). Recall is worth far
   more than precision here.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

# PDF page 6: "A fixed penalty of PENALTY = n_risk = 30"
PENALTY = 30.0

# PDF page 3: onset_day_offset is an integer 1..30; recovery_duration observed 5..20
ONSET_RANGE = (1, 30)
RECOVERY_RANGE = (5, 20)


def baseline_mae(true_vals, train_mean):
    """MAE of the organisers' baseline: the training-set MEAN, constant for
    every injured athlete. Note MEAN, not median -- the PDF is explicit."""
    true_vals = np.asarray(true_vals, dtype=float)
    return float(np.abs(true_vals - float(train_mean)).mean())


def skill(mae_model, mae_baseline):
    """skill = max(0, 1 - MAE_model / MAE_baseline)"""
    if mae_baseline <= 0:
        return 0.0
    return float(max(0.0, 1.0 - mae_model / mae_baseline))


def task_b_mae(hit_mask, abs_err):
    """MAE over the truly-injured athletes, applying PENALTY where missed.

    hit_mask : bool array over truly-injured athletes; True where we predicted 1
    abs_err  : |pred - true| over those same athletes (value ignored where missed)
    """
    hit_mask = np.asarray(hit_mask, dtype=bool)
    abs_err = np.asarray(abs_err, dtype=float)
    return float(np.where(hit_mask, abs_err, PENALTY).mean())


def official_score(y_true, y_pred,
                   onset_true, onset_pred,
                   rec_true, rec_pred,
                   train_mean_onset, train_mean_rec):
    """Full Task A + Task B panel.

    y_true / y_pred    : binary injured_in_risk_window over ALL athletes
    onset_* / rec_*     : arrays over the TRULY INJURED athletes only, aligned
                          to y_true == 1 in order
    train_mean_*        : the training-set means defining the baseline
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    inj = y_true == 1

    hit = y_pred[inj] == 1
    err_on = np.abs(np.asarray(onset_pred, dtype=float) - np.asarray(onset_true, dtype=float))
    err_rc = np.abs(np.asarray(rec_pred, dtype=float) - np.asarray(rec_true, dtype=float))

    mae_on = task_b_mae(hit, err_on)
    mae_rc = task_b_mae(hit, err_rc)
    base_on = baseline_mae(onset_true, train_mean_onset)
    base_rc = baseline_mae(rec_true, train_mean_rec)

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    s_on = skill(mae_on, base_on)
    s_rc = skill(mae_rc, base_rc)

    return {
        "f1": f1,
        "recall_injured": float(hit.mean()) if inj.any() else 0.0,
        "precision": float((y_true[y_pred == 1] == 1).mean()) if (y_pred == 1).any() else 0.0,
        "positive_rate": float(y_pred.mean()),
        "mae_onset_model": mae_on,
        "mae_onset_baseline": base_on,
        "skill_onset": s_on,
        "mae_recovery_model": mae_rc,
        "mae_recovery_baseline": base_rc,
        "skill_recovery": s_rc,
        # the PDF gives no Task A / Task B weighting; report the balanced mean
        # and let the sensitivity analysis in RESULTS.md carry the caveat
        "mean_of_three": float((f1 + s_on + s_rc) / 3.0),
    }


def _self_check():
    """Adversarial checks on the two rules that are easy to implement backwards."""
    rng = np.random.default_rng(0)
    n, n_inj = 400, 140
    y = np.zeros(n, dtype=int)
    y[:n_inj] = 1
    on_true = rng.integers(1, 31, n_inj).astype(float)
    rc_true = rng.integers(5, 21, n_inj).astype(float)
    mo, mr = on_true.mean(), rc_true.mean()

    # 1. Predicting nobody injured -> every injured athlete is a miss -> MAE == PENALTY.
    #    PENALTY(30) far exceeds either baseline, so both skills must clamp to 0.
    r = official_score(y, np.zeros(n, int), on_true, on_true, rc_true, rc_true, mo, mr)
    assert r["recall_injured"] == 0.0, r
    assert r["mae_onset_model"] == PENALTY and r["mae_recovery_model"] == PENALTY, r
    assert r["skill_onset"] == 0.0 and r["skill_recovery"] == 0.0, r

    # 2. Predicting everyone injured -> no misses -> recall 1, MAE is the raw error.
    #    Perfect timing predictions -> MAE 0 -> skill exactly 1.
    r = official_score(y, np.ones(n, int), on_true, on_true, rc_true, rc_true, mo, mr)
    assert r["recall_injured"] == 1.0, r
    assert r["mae_onset_model"] == 0.0 and r["skill_onset"] == 1.0, r
    assert r["skill_recovery"] == 1.0, r

    # 3. A false positive must not affect Task B at all (it is not truly injured).
    p_clean = np.zeros(n, int); p_clean[:n_inj] = 1
    p_fp = p_clean.copy(); p_fp[n_inj:] = 1          # flag every healthy athlete too
    a = official_score(y, p_clean, on_true, on_true, rc_true, rc_true, mo, mr)
    b = official_score(y, p_fp, on_true, on_true, rc_true, rc_true, mo, mr)
    assert a["skill_onset"] == b["skill_onset"], (a, b)
    assert a["skill_recovery"] == b["skill_recovery"], (a, b)
    assert b["f1"] < a["f1"], "false positives must cost Task A precision"

    # 4. The baseline itself must score skill exactly 0 (MAE_model == MAE_baseline).
    r = official_score(y, np.ones(n, int),
                       on_true, np.full(n_inj, mo), rc_true, np.full(n_inj, mr), mo, mr)
    assert abs(r["skill_onset"]) < 1e-12, r
    assert abs(r["skill_recovery"]) < 1e-12, r

    # 5. Break-even recall. With a per-hit error of `e`, skill stays 0 while
    #    R*e + (1-R)*PENALTY >= baseline, i.e. R <= (PENALTY - base) / (PENALTY - e).
    #    Checked against our own measured hit-MAEs (onset ~2.62, recovery ~2.93),
    #    which is what makes the bar so high: ~0.82 onset, ~0.99 recovery.
    base_on_, base_rc_ = baseline_mae(on_true, mo), baseline_mae(rc_true, mr)
    for e, base, name in [(2.62, base_on_, "onset"), (2.93, base_rc_, "recovery")]:
        r_star = (PENALTY - base) / (PENALTY - e)          # break-even recall
        for R, expect_zero in [(r_star - 0.02, True), (min(r_star + 0.02, 1.0), False)]:
            hit = np.zeros(n_inj, dtype=bool)
            hit[: int(round(R * n_inj))] = True
            s = skill(task_b_mae(hit, np.full(n_inj, e)), base)
            assert (s == 0.0) is expect_zero, (name, R, r_star, s)
    # 6. The headline claim is about the REAL label distribution, not synthetic
    #    uniforms. Real recovery_duration is concentrated (mode 8-15) so its
    #    baseline is only ~3.24 and it needs near-perfect recall; onset is
    #    near-uniform over 1..30 so its baseline ~7.61 tolerates ~18% misses.
    from pathlib import Path
    labels = Path(__file__).resolve().parents[1] / "data" / "train" / "train_labels.csv"
    if labels.exists():
        import pandas as pd
        d = pd.read_csv(labels).query("injured_in_risk_window == 1")
        b_on = baseline_mae(d.onset_day_offset, d.onset_day_offset.mean())
        b_rc = baseline_mae(d.recovery_duration, d.recovery_duration.mean())
        be_on = (PENALTY - b_on) / (PENALTY - 2.62)
        be_rc = (PENALTY - b_rc) / (PENALTY - 2.93)
        assert 0.78 < be_on < 0.86, (b_on, be_on)
        assert be_rc > 0.97, (b_rc, be_rc)
        print("  real baselines      : onset %.4f  recovery %.4f" % (b_on, b_rc))
        print("  break-even recall   : onset %.4f  recovery %.4f" % (be_on, be_rc))
    else:
        print("  (train_labels.csv absent -- skipped real-data break-even check)")

    print("score.py self-check passed (5 checks)")


if __name__ == "__main__":
    _self_check()
