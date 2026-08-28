"""
The ceiling under the ORGANISERS' metric, not under AUC.

AUC is nowhere in the scoring rule -- it was only ever our internal ranking
diagnostic. What is actually scored (PDF p.6, see src/score.py) is:

    Task A   F1
    Task B   skill_onset, skill_recovery, with a flat PENALTY=30 for a
             missed injury and skill = max(0, 1 - MAE_model/MAE_baseline)

so the honest question is "how close are we to the best achievable F1 and skill
scores", and `oracle.py`'s F1 ceiling does not answer it. That number is the F1
at the *F1-optimal* threshold. But an oracle scored on this metric would not sit
at that threshold either: it faces exactly the same asymmetry we do -- a miss
costs 30 against baselines of ~7.6 and ~3.2 -- so it would also drop its
threshold and trade F1 for skill.

This script therefore sweeps the ORACLE's threshold over the real objective and
reports the best it can do, giving a like-for-like ceiling for every component.

The oracle here is the same construction as oracle.py: a cross-fitted isotonic
map from the recovered load-ramp latent to a true injury probability p(x), plus
conditional-median timing predictions whose error is the irreducible conditional
MAD. Expectations are taken under p rather than the realised labels, so the
numbers are the expected score of a perfectly-calibrated model, not a lucky draw.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import PENALTY
from oracle import load, crossfit_isotonic
from final import conditional_mad_ceiling

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
GRID = np.linspace(0.0, 0.95, 192)


def expected_panel(p, thr, mad_on, mad_rc, base_on, base_rc):
    """Expected F1 and skills at threshold `thr`, in expectation over p.

    Every athlete contributes probabilistically: an athlete with probability p_i
    is truly injured with that probability, so the expected count of injured
    athletes we catch is sum(p_i * flagged_i), and the expected Task B error is
    the conditional MAD where flagged and PENALTY where not.
    """
    pred = p >= thr
    tp = (p * pred).sum()
    fp = ((1 - p) * pred).sum()
    fn = (p * ~pred).sum()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)

    # Task B population is the truly-injured athletes, weighted by p
    w = p / p.sum()
    err_on = np.where(pred, mad_on, PENALTY)
    err_rc = np.where(pred, mad_rc, PENALTY)
    mae_on = float((w * err_on).sum())
    mae_rc = float((w * err_rc).sum())
    return {
        "f1": float(f1),
        "recall": float((p * pred).sum() / p.sum()),
        "positive_rate": float(pred.mean()),
        "mae_onset": mae_on,
        "mae_recovery": mae_rc,
        "skill_onset": float(max(0.0, 1 - mae_on / base_on)),
        "skill_recovery": float(max(0.0, 1 - mae_rc / base_rc)),
    }


def main():
    df = load()
    y = df["injured_in_risk_window"]
    inj = (y == 1).to_numpy()

    # p = our own OOF scores, cross-fit-calibrated to probabilities.
    #
    # NOT oracle.py's latent-based p. That construction conditions on a single
    # recovered dimension (hr_acwr) and is measurably WEAKER than the deployed
    # model (latent AUC 0.7583 vs model 0.7625), so using it as a ceiling
    # produced the nonsense of the model scoring 113% of "maximum". A ceiling
    # has to condition on at least as much information as the model it bounds.
    oof = np.load(REPORTS / "oof_blend.npy")
    p = np.asarray(crossfit_isotonic(oof, y), dtype=float)

    # organisers' baseline: the training-set MEAN, for every injured athlete
    y_on = df.loc[inj, "onset_day_offset"].reset_index(drop=True).to_numpy(float)
    y_rc = df.loc[inj, "recovery_duration"].reset_index(drop=True).to_numpy(float)
    base_on = float(np.abs(y_on - y_on.mean()).mean())
    base_rc = float(np.abs(y_rc - y_rc.mean()).mean())

    # Irreducible timing error: spread remaining inside groups of athletes the
    # model scores identically (final.py's estimator, conditioned on all 35
    # features rather than one latent).
    on_oof = np.load(REPORTS / "oof_onset_day_offset.npy")
    rc_oof = np.load(REPORTS / "final_oof_recovery.npy")
    mad_on = np.full(len(p), conditional_mad_ceiling(on_oof, y_on))
    mad_rc = np.full(len(p), conditional_mad_ceiling(rc_oof, y_rc))

    print("\nbaselines (training-set MEAN): onset %.4f  recovery %.4f" % (base_on, base_rc))
    print("irreducible timing error (model-conditioned): onset %.4f  recovery %.4f"
          % (mad_on[0], mad_rc[0]))
    print("  (oracle.py's latent-conditioned onset figure, 3.2065, is far looser"
          " -- it knows only hr_acwr)")

    rows = [(t, expected_panel(p, t, mad_on, mad_rc, base_on, base_rc)) for t in GRID]

    print("\n  thr   posrate  recall     F1    skill_on  skill_rc   mean3")
    for t, r in rows[::12]:
        m3 = (r["f1"] + r["skill_onset"] + r["skill_recovery"]) / 3
        print("  %.2f   %.4f  %.4f  %.4f   %.4f    %.4f   %.4f" % (
            t, r["positive_rate"], r["recall"], r["f1"],
            r["skill_onset"], r["skill_recovery"], m3))

    best_t, best = max(rows, key=lambda kv: (kv[1]["f1"] + kv[1]["skill_onset"]
                                             + kv[1]["skill_recovery"]) / 3)
    f1_t, f1_best = max(rows, key=lambda kv: kv[1]["f1"])
    m3 = (best["f1"] + best["skill_onset"] + best["skill_recovery"]) / 3

    print("\nORACLE, F1-optimal threshold %.3f:" % f1_t)
    print("  F1 %.4f   skill_on %.4f   skill_rc %.4f   mean3 %.4f" % (
        f1_best["f1"], f1_best["skill_onset"], f1_best["skill_recovery"],
        (f1_best["f1"] + f1_best["skill_onset"] + f1_best["skill_recovery"]) / 3))
    print("ORACLE, metric-optimal threshold %.3f:" % best_t)
    print("  F1 %.4f   skill_on %.4f   skill_rc %.4f   mean3 %.4f" % (
        best["f1"], best["skill_onset"], best["skill_recovery"], m3))
    print("  -> even the oracle abandons the F1 optimum: it gives up %.4f F1"
          % (f1_best["f1"] - best["f1"]))
    print("     to gain %.4f skill, exactly the trade we make."
          % ((best["skill_onset"] + best["skill_recovery"])
             - (f1_best["skill_onset"] + f1_best["skill_recovery"])))

    out = {
        "oracle_official_threshold": float(best_t),
        "oracle_official_f1": best["f1"],
        "oracle_official_skill_onset": best["skill_onset"],
        "oracle_official_skill_recovery": best["skill_recovery"],
        "oracle_official_mean_of_three": float(m3),
        "oracle_official_recall": best["recall"],
        "oracle_f1_optimal_threshold": float(f1_t),
        "oracle_f1_at_f1_optimum": f1_best["f1"],
        "oracle_skill_onset_at_f1_optimum": f1_best["skill_onset"],
        "oracle_skill_recovery_at_f1_optimum": f1_best["skill_recovery"],
        "mae_onset_baseline": base_on,
        "mae_recovery_baseline": base_rc,
        "oracle_irreducible_mae_onset": float(mad_on[0]),
        "oracle_irreducible_mae_recovery": float(mad_rc[0]),
    }
    (REPORTS / "official_ceiling.json").write_text(json.dumps(out, indent=2))

    mp = REPORTS / "metrics.json"
    if mp.exists():
        M = json.loads(mp.read_text())
        M.update(out)
        mp.write_text(json.dumps(M, indent=2))
        print("\n  ours vs ceiling:")
        over = []
        for k, ck, label in [("official_f1", "oracle_official_f1", "F1"),
                             ("official_skill_onset", "oracle_official_skill_onset", "skill_onset"),
                             ("official_skill_recovery", "oracle_official_skill_recovery", "skill_recovery"),
                             ("official_mean_of_three", "oracle_official_mean_of_three", "mean3")]:
            if k in M:
                ceil = out[ck]
                pct = 100 * M[k] / ceil if ceil else float("nan")
                print("    %-15s %.4f  vs %.4f  (%.1f%%)" % (label, M[k], ceil, pct))
                if pct > 101.0:
                    over.append(label)
        if over:
            print("\n  WARNING: exceeds ceiling on " + ", ".join(over) + "."
                  " A ceiling is an ESTIMATE of irreducible error, not a proven"
                  " bound -- exceeding it means the estimate is too pessimistic"
                  " (usually: it conditions on less information than the model).")
    print("\nwrote reports/official_ceiling.json")


if __name__ == "__main__":
    main()
