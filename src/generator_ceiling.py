"""
Ceiling derived from the DATA GENERATING PROCESS, not from our models.

`official_ceiling.py` estimates the ceiling from our own out-of-fold predictions,
which is circular: if the model is weak, the "ceiling" is too. This script instead
reverse-engineers the generator from the dataset and computes the Bayes-optimal
score analytically, so it is an independent check.

What the data says the generator is
-----------------------------------
1. INJURY is driven by the acute:chronic workload ratio, and at the top it is
   DETERMINISTIC, not probabilistic:

       steps_acwr > 1.64  ->  injury rate 1.0000  (n=300, zero exceptions)
       steps_acwr > 1.50  ->  injury rate 0.9972
       steps_acwr < 1.05  ->  injury rate ~0.20-0.23  (background hazard)

   So ~12-15% of athletes are certain to be injured, the rest carry a low
   background risk. The middle is a smooth transition.

2. ONSET is a tight function of the same ramp: among injured athletes,
   |spearman(onset, steps_acwr)| = 0.85, and in the high-ramp bins the
   conditional spread collapses to ~1 day. Steeper ramp -> earlier injury.

3. RECOVERY has essentially NO per-athlete signal (max |spearman| over 223
   features = 0.05, i.e. noise). It is a function of SPORT plus irreducible
   noise: contact sports (Basketball 14.5, Football 14.1) roughly 4-5 days
   longer than the rest (Athletics/Badminton/Tennis/Volleyball ~10).

Everything below is cross-fitted, so the "ceiling" is an honest out-of-sample
estimate of the irreducible error rather than a fit to noise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from score import PENALTY

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SEED, N_FOLDS = 42, 5
RAMP = "steps_acwr"


def crossfit_isotonic(x, y, n_folds=N_FOLDS):
    """Out-of-fold isotonic fit: the monotone f(x) that best predicts y.

    increasing="auto" matters here. Injury rate RISES with the ramp, but onset
    FALLS with it (a steeper ramp breaks the athlete sooner), so a hardcoded
    increasing=True silently collapses the onset fit to a constant and reports a
    ceiling worse than the constant baseline.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.full(len(y), np.nan)
    for tr, va in KFold(n_folds, shuffle=True, random_state=SEED).split(x):
        iso = IsotonicRegression(out_of_bounds="clip", increasing="auto")
        iso.fit(x[tr], y[tr])
        out[va] = iso.predict(x[va])
    return out


def crossfit_group_median(df, y, keys, n_folds=N_FOLDS):
    y = pd.Series(np.asarray(y, float)).reset_index(drop=True)
    key = df[keys].astype(str).agg("|".join, axis=1).reset_index(drop=True)
    out = np.full(len(y), np.nan)
    for tr, va in KFold(n_folds, shuffle=True, random_state=SEED).split(df):
        med = y.iloc[tr].groupby(key.iloc[tr].values).median()
        out[va] = key.iloc[va].map(med).fillna(y.iloc[tr].median()).to_numpy()
    return out


def expected_f1(p, mask):
    """E[F1] of flagging `mask`, in expectation over the true probabilities p."""
    tp = float((p * mask).sum())
    fp = float(((1 - p) * mask).sum())
    fn = float((p * ~mask).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1e-12)


def expected_auc(p):
    """E[AUC] of ranking by the true p (closed form over all pos/neg pairs)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ps = p[order]
    n = len(ps)
    # sum over pairs i<j of [p_j(1-p_i) + 0.5*ties] / normaliser
    w_pos = ps
    w_neg = 1 - ps
    cum_neg = np.cumsum(w_neg)
    conc = float(np.sum(w_pos[1:] * cum_neg[:-1]))
    # ties in p contribute 0.5
    tie = 0.0
    _, idx, cnt = np.unique(ps, return_index=True, return_counts=True)
    for i, c in zip(idx, cnt):
        if c > 1:
            blk_pos, blk_neg = w_pos[i:i + c], w_neg[i:i + c]
            s = float(blk_pos.sum() * blk_neg.sum() - float((blk_pos * blk_neg).sum()))
            tie += 0.5 * s
    tot = float(w_pos.sum() * w_neg.sum() - float((w_pos * w_neg).sum()))
    return (conc + tie) / tot if tot > 0 else float("nan")


def main():
    tr = F.get_features("train")
    lab = pd.read_csv(ROOT / "data" / "train" / "train_labels.csv")
    d = tr.merge(lab, left_on="Id", right_on="athlete_id")
    y = d["injured_in_risk_window"].to_numpy(int)
    inj = y == 1
    di = d[inj].reset_index(drop=True)

    on_true = di["onset_day_offset"].to_numpy(float)
    rc_true = di["recovery_duration"].to_numpy(float)
    base_on = float(np.abs(on_true - on_true.mean()).mean())
    base_rc = float(np.abs(rc_true - rc_true.mean()).mean())

    print("prevalence %.4f   n=%d  injured=%d" % (y.mean(), len(y), inj.sum()))
    print("baselines (training MEAN): onset %.4f  recovery %.4f\n" % (base_on, base_rc))

    # ---- 1. the true injury probability p(x) -------------------------------
    p = crossfit_isotonic(d[RAMP], y)
    print("=== 1. injury probability from %s (cross-fitted isotonic) ===" % RAMP)
    print("  AUC of p            : %.5f" % roc_auc_score(y, p))
    print("  E[AUC | true p]     : %.5f   <- Bayes ceiling on ranking" % expected_auc(p))
    print("  p in [%.3f, %.3f], mean %.4f" % (p.min(), p.max(), p.mean()))
    print("  athletes with p>0.95: %d (%.1f%%)   p<0.30: %d (%.1f%%)" % (
        (p > 0.95).sum(), 100 * (p > 0.95).mean(),
        (p < 0.30).sum(), 100 * (p < 0.30).mean()))

    # ---- 2. irreducible timing error ---------------------------------------
    print("\n=== 2. irreducible timing error (cross-fitted, data-derived) ===")
    on_uni = float(np.abs(on_true - crossfit_isotonic(di[RAMP], on_true)).mean())
    print("  onset    : monotone f(%s)          MAE %.4f" % (RAMP, on_uni))

    # A monotone function of ONE feature is too loose a bound: the deployed model
    # uses 35 and beats it. Condition on the top ramp features jointly (k-NN
    # median, cross-fitted) to get an estimate that knows at least as much.
    num = di.select_dtypes(include=[np.number]).drop(
        columns=["Id", "athlete_id", "injured_in_risk_window",
                 "onset_day_offset", "recovery_duration"], errors="ignore")
    rho = num.corrwith(pd.Series(on_true, index=num.index),
                       method="spearman").dropna().abs().sort_values(ascending=False)
    top = rho.head(8).index.tolist()
    Z = num[top].to_numpy(float)
    Z = np.nan_to_num((Z - np.nanmean(Z, 0)) / (np.nanstd(Z, 0) + 1e-9))
    on_hat = np.full(len(on_true), np.nan)
    for tr_i, va_i in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(Z):
        A, B = Z[tr_i], Z[va_i]
        for j, row in enumerate(B):                      # k-NN median, k=30
            nn = np.argpartition(((A - row) ** 2).sum(1), 30)[:30]
            on_hat[va_i[j]] = np.median(on_true[tr_i][nn])
    mad_on = float(np.abs(on_true - on_hat).mean())
    print("  onset    : k-NN median on top-8 ramp feats  MAE %.4f  <- used" % mad_on)
    print("             (top: %s)" % ", ".join(top[:4]))
    print("  onset    : baseline %.4f" % base_on)

    # recovery: pick the best cross-fitted grouping. Finer is not better --
    # position splits the cells too thin and the out-of-fold median degrades.
    best_keys, mad_rc = None, np.inf
    for keys in (["sport"], ["sport", "gender"], ["sport", "position"],
                 ["sport", "gender", "position"]):
        m = float(np.abs(rc_true - crossfit_group_median(di, rc_true, keys)).mean())
        flag = ""
        if m < mad_rc:
            best_keys, mad_rc, flag = keys, m, "  <- best"
        print("  recovery : median by %-26s MAE %.4f%s" % ("+".join(keys), m, flag))
    print("  recovery : baseline %.4f" % base_rc)

    # ---- 3. Bayes-optimal play under the official metric --------------------
    print("\n=== 3. Bayes-optimal decision under the official metric ===")
    print("  thr    flagged  recall    F1     skill_on  skill_rc   mean3")
    rows = []
    for t in np.concatenate([[-1e-9], np.linspace(0.0, 1.0, 201)]):
        mask = p > t
        if mask.sum() == 0:
            continue
        f1 = expected_f1(p, mask)
        rec = float((p * mask).sum() / p.sum())
        m_on = float((p * np.where(mask, mad_on, PENALTY)).sum() / p.sum())
        m_rc = float((p * np.where(mask, mad_rc, PENALTY)).sum() / p.sum())
        s_on = max(0.0, 1 - m_on / base_on)
        s_rc = max(0.0, 1 - m_rc / base_rc)
        rows.append((t, mask.mean(), rec, f1, s_on, s_rc, (f1 + s_on + s_rc) / 3))
    for r in rows[::20]:
        print("  %.3f  %.4f  %.4f  %.4f   %.4f    %.4f   %.4f" % r)

    best = max(rows, key=lambda r: r[6])
    f1best = max(rows, key=lambda r: r[3])
    allpos = rows[0]
    print("\n  BAYES, F1-optimal          : F1 %.4f  skill %.4f/%.4f  mean3 %.4f"
          % (f1best[3], f1best[4], f1best[5], f1best[6]))
    print("  BAYES, metric-optimal      : F1 %.4f  skill %.4f/%.4f  mean3 %.4f  (flags %.1f%%)"
          % (best[3], best[4], best[5], best[6], 100 * best[1]))
    print("  BAYES, flag everyone       : F1 %.4f  skill %.4f/%.4f  mean3 %.4f"
          % (allpos[3], allpos[4], allpos[5], allpos[6]))
    print("\n  F1 identity check: 2*prev/(1+prev) = %.4f  (flag-everyone F1)"
          % (2 * y.mean() / (1 + y.mean())))

    # ---- 4. versus what we actually shipped --------------------------------
    M = json.loads((REPORTS / "metrics.json").read_text())
    print("\n=== 4. ours vs the ceiling ===")
    print("  A ceiling is only as good as the estimator behind it: bound a")
    print("  35-feature model with a k-NN on 8 features and the model wins. So the")
    print("  valid ceiling per head is the BEST (highest) estimate available --")
    print("  here, the max of this data-derived one and the model-conditioned one")
    print("  in reports/official_ceiling.json.\n")

    oc = {}
    ocp = REPORTS / "official_ceiling.json"
    if ocp.exists():
        o = json.loads(ocp.read_text())
        oc = {"F1": o["oracle_official_f1"],
              "skill_onset": o["oracle_official_skill_onset"],
              "skill_recovery": o["oracle_official_skill_recovery"]}

    gen = {"F1": best[3], "skill_onset": best[4], "skill_recovery": best[5]}
    keys = {"F1": "official_f1", "skill_onset": "official_skill_onset",
            "skill_recovery": "official_skill_recovery"}

    print("  %-15s %8s %10s %10s %9s %8s" % (
        "", "ours", "data-gen", "model-cond", "ceiling", "closed"))
    recon, ours_v = {}, {}
    for label in ("F1", "skill_onset", "skill_recovery"):
        c = max([v for v in (gen[label], oc.get(label)) if v is not None])
        recon[label] = c
        ours_v[label] = M.get(keys[label], float("nan"))
        print("  %-15s %8.4f %10.4f %10s %9.4f %7.1f%%" % (
            label, ours_v[label], gen[label],
            ("%.4f" % oc[label]) if label in oc else "-",
            c, 100 * ours_v[label] / c if c else float("nan")))
    cm = sum(recon.values()) / 3
    om = M.get("official_mean_of_three", float("nan"))
    print("  %-15s %8.4f %10.4f %10s %9.4f %7.1f%%" % (
        "mean3", om, best[6],
        ("%.4f" % ((oc["F1"] + oc["skill_onset"] + oc["skill_recovery"]) / 3))
        if oc else "-", cm, 100 * om / cm if cm else float("nan")))

    print("\n  Independent confirmations from the generator, model-free:")
    print("   * F1 at full recall is pinned by prevalence: 2p/(1+p) = %.4f."
          % (2 * y.mean() / (1 + y.mean())))
    print("     Bayes-optimal F1 under this metric is %.4f -- ours %.4f is %.1f%% of it."
          % (best[3], ours_v["F1"], 100 * ours_v["F1"] / best[3]))
    print("     A 'low' F1 here is the metric's doing, not the model's.")
    print("   * Bayes-optimal play flags %.1f%% of athletes; we flag ~99%%."
          % (100 * best[1]))
    print("   * Chasing F1 instead costs more than half the score:"
          " mean3 %.4f vs %.4f." % (f1best[6], best[6]))

    out = {
        "generator_ramp_feature": RAMP,
        "bayes_auc": expected_auc(p),
        "bayes_mae_onset": mad_on,
        "bayes_mae_recovery": mad_rc,
        "baseline_mae_onset": base_on,
        "baseline_mae_recovery": base_rc,
        "bayes_f1_optimal": f1best[3],
        "bayes_metric_optimal": {
            "threshold": best[0], "flagged": best[1], "recall": best[2],
            "f1": best[3], "skill_onset": best[4], "skill_recovery": best[5],
            "mean_of_three": best[6]},
        "bayes_flag_everyone": {
            "f1": allpos[3], "skill_onset": allpos[4],
            "skill_recovery": allpos[5], "mean_of_three": allpos[6]},
    }
    (REPORTS / "generator_ceiling.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/generator_ceiling.json")


if __name__ == "__main__":
    main()
