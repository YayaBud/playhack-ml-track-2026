"""
Choose the classification threshold against the organisers' real metric.

Everything before this tuned the threshold for F1 alone, because the metric had
not been located. The PDF (page 6, see src/score.py) changes the objective:

  * Task B is scored over the TRULY injured athletes. A false positive is not in
    that population at all, so it costs Task B nothing -- only Task A precision.
  * A false negative on a truly injured athlete costs a flat PENALTY = 30 on
    BOTH timing heads, against baselines of only ~7.61 (onset) and ~3.24
    (recovery).

So recall is worth far more than precision, and the F1-optimal threshold (0.40,
recall 0.51) scores exactly 0.0 on both skill components. Break-even recall is
0.818 for onset and 0.988 for recovery -- recovery is knife-edge.

That asymmetry drives the selection rule below. The naive argmax over the sweep
excludes ~0.9% of athletes for +0.004 F1, but if that exclusion clips even a
handful of genuinely injured test athletes, recovery skill collapses from ~0.10
to 0. Expected-value-wise the upside is ~0.004 and the downside ~0.10, so this
script deliberately takes a *fraction* of the maximum safe exclusion rather than
the knife edge, and applies it as a rank quantile (robust to the OOF-vs-test
distribution shift that bit the earlier F1 threshold).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import official_score, PENALTY, ONSET_RANGE, RECOVERY_RANGE

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

SAFETY_FRACTION = 0.5     # use half the maximum OOF-safe exclusion
MAX_EXCLUSION = 0.02      # never drop more than 2% of athletes


def main():
    lab = pd.read_csv(ROOT / "data" / "train" / "train_labels.csv")
    y = lab.injured_in_risk_window.to_numpy()
    inj = y == 1
    on_true = lab.loc[inj, "onset_day_offset"].to_numpy(float)
    rc_true = lab.loc[inj, "recovery_duration"].to_numpy(float)
    mo, mr = on_true.mean(), rc_true.mean()

    oof = np.load(REPORTS / "oof_blend.npy")
    on_oof = np.clip(np.load(REPORTS / "oof_onset_day_offset.npy"), *ONSET_RANGE)
    rc_oof = np.clip(np.load(REPORTS / "final_oof_recovery.npy"), *RECOVERY_RANGE)

    def panel(pred):
        return official_score(y, pred, on_true, on_oof, rc_true, rc_oof, mo, mr)

    print("Baselines (PDF: training-set MEAN): onset %.4f  recovery %.4f" % (
        np.abs(on_true - mo).mean(), np.abs(rc_true - mr).mean()))
    print("PENALTY = %.0f\n" % PENALTY)

    print("  thr   posrate  recall     F1    skill_on  skill_rc   mean3")
    for t in np.arange(0.0, 1.0, 0.05):
        r = panel((oof >= t).astype(int))
        print("  %.2f   %.4f  %.4f  %.4f   %.4f    %.4f   %.4f" % (
            t, r["positive_rate"], r["recall_injured"], r["f1"],
            r["skill_onset"], r["skill_recovery"], r["mean_of_three"]))

    # --- how much can we safely exclude? -----------------------------------
    # Any athlete scoring below the lowest-scoring truly-injured athlete can be
    # dropped without losing recall ON OOF. That is the ceiling, not the target.
    min_inj_score = oof[inj].min()
    max_safe_excl = float((oof < min_inj_score).mean())
    all_ones = panel(np.ones_like(y))
    print("\nlowest OOF score among truly-injured athletes : %.4f" % min_inj_score)
    print("max exclusion keeping OOF recall 1.0          : %.4f" % max_safe_excl)
    print("all-ones baseline: F1 %.4f  skill_on %.4f  skill_rc %.4f  mean3 %.4f" % (
        all_ones["f1"], all_ones["skill_onset"], all_ones["skill_recovery"],
        all_ones["mean_of_three"]))

    excl = min(max_safe_excl * SAFETY_FRACTION, MAX_EXCLUSION)
    thr_oof = float(np.quantile(oof, excl)) if excl > 0 else 0.0
    chosen = panel((oof >= thr_oof).astype(int))
    print("\nchosen exclusion %.4f (%.0f%% of the safe ceiling, capped at %.0f%%)" % (
        excl, SAFETY_FRACTION * 100, MAX_EXCLUSION * 100))
    print("  OOF: thr %.4f  posrate %.4f  recall %.4f  F1 %.4f"
          "  skill_on %.4f  skill_rc %.4f  mean3 %.4f" % (
              thr_oof, chosen["positive_rate"], chosen["recall_injured"],
              chosen["f1"], chosen["skill_onset"], chosen["skill_recovery"],
              chosen["mean_of_three"]))
    assert chosen["recall_injured"] == 1.0, "selection rule must preserve OOF recall"
    print("  gain over all-ones: F1 %+.4f  mean3 %+.4f" % (
        chosen["f1"] - all_ones["f1"],
        chosen["mean_of_three"] - all_ones["mean_of_three"]))

    # --- sensitivity to the unknown Task A / Task B weighting --------------
    f1_best_t, f1_best = max(
        ((t, panel((oof >= t).astype(int))) for t in np.arange(0.05, 0.96, 0.01)),
        key=lambda kv: kv[1]["f1"])
    print("\nsensitivity (Task A weight w, score = w*F1 + (1-w)*mean(skills)):")
    print("   w     F1-optimal thr %.2f      our choice      winner" % f1_best_t)
    for w in [0.25, 0.5, 0.75, 0.8, 0.9]:
        a = w * f1_best["f1"] + (1 - w) * (f1_best["skill_onset"] + f1_best["skill_recovery"]) / 2
        b = w * chosen["f1"] + (1 - w) * (chosen["skill_onset"] + chosen["skill_recovery"]) / 2
        print("  %.2f        %.4f              %.4f          %s" % (
            w, a, b, "ours" if b >= a else "F1-threshold"))

    # --- apply to test ------------------------------------------------------
    # Rank quantile, not the absolute OOF probability: OOF rows are scored by
    # fold-models averaged over 3 seeds while test rows come from single
    # full-data fits, so the two probability distributions are not on the same
    # scale (this mismatch is what made the earlier F1 threshold fire on 25% of
    # test when it should have fired on ~38%).
    p_test = np.load(REPORTS / "final_test_clf.npy")
    on_test = np.clip(np.load(REPORTS / "final_test_onset.npy"), *ONSET_RANGE)
    rc_test = np.clip(np.load(REPORTS / "final_test_recovery.npy"), *RECOVERY_RANGE)
    thr_test = float(np.quantile(p_test, excl)) if excl > 0 else -np.inf
    pred_test = (p_test >= thr_test).astype(int)
    print("\ntest: thr %.4f  positive rate %.4f  (excluded %d of %d)" % (
        thr_test, pred_test.mean(), (pred_test == 0).sum(), len(pred_test)))

    # the prediction arrays are in feature-table order; assert that really is
    # sample_submission order before pairing them with ids
    import features as F
    ids = F.get_features("test")["Id"].to_numpy()
    ss_ids = pd.read_csv(ROOT / "data" / "test" / "sample_submission.csv")["athlete_id"].to_numpy()
    assert np.array_equal(ids, ss_ids), "feature order does not match sample_submission order"

    sub = pd.DataFrame({
        "athlete_id": ids,
        "injured_in_risk_window": pred_test,
        "onset_day_offset": np.round(on_test).astype(int),
        "recovery_duration": np.round(rc_test).astype(int),
    })
    sub.to_csv(ROOT / "submission.csv", index=False)

    M = json.loads((REPORTS / "metrics.json").read_text())
    M.update({
        "official_f1": chosen["f1"],
        "official_skill_onset": chosen["skill_onset"],
        "official_skill_recovery": chosen["skill_recovery"],
        "official_mean_of_three": chosen["mean_of_three"],
        "official_mae_onset_model": chosen["mae_onset_model"],
        "official_mae_onset_baseline": chosen["mae_onset_baseline"],
        "official_mae_recovery_model": chosen["mae_recovery_model"],
        "official_mae_recovery_baseline": chosen["mae_recovery_baseline"],
        "official_oof_recall": chosen["recall_injured"],
        "official_exclusion_fraction": excl,
        "official_max_safe_exclusion": max_safe_excl,
        "threshold_oof": thr_oof,
        "threshold_test": thr_test,
        "test_positive_rate": float(pred_test.mean()),
        "f1_optimal_threshold_would_score": {
            "threshold": float(f1_best_t), "f1": f1_best["f1"],
            "skill_onset": f1_best["skill_onset"],
            "skill_recovery": f1_best["skill_recovery"],
        },
        "penalty": PENALTY,
    })
    (REPORTS / "metrics.json").write_text(json.dumps(M, indent=2))

    mf = MODELS / "manifest.json"
    if mf.exists():
        man = json.loads(mf.read_text())
        man["threshold"] = {"mode": "rank_quantile", "exclusion_fraction": excl}
        mf.write_text(json.dumps(man, indent=2))

    print("\nwrote submission.csv + reports/metrics.json + models/manifest.json")


if __name__ == "__main__":
    main()
