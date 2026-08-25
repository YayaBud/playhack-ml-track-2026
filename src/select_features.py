"""
Feature-count sweep.

The 206-column matrix is mostly redundant: the ACWR/ramp family members are
r ~ 0.99 with each other, so a tree ensemble spreads its splits across near-
duplicate columns and ends up BELOW a single raw `hr_acwr` (AUC 0.7583). This
script finds how many features each head actually wants, ranking by permutation
-free gain importance computed inside the CV loop (so selection never sees the
held-out fold).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_absolute_error
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 202]
N_FOLDS = 5
KS = [3, 5, 8, 12, 20, 35, 60, 100, 206]


def load_xy():
    tr = F.get_features("train")
    te = F.get_features("test")
    lab = pd.read_csv(F.DATA / "train" / "train_labels.csv")
    tr = tr.merge(lab, left_on="Id", right_on="athlete_id")
    drop = {"athlete_id", "injured_in_risk_window", "onset_day_offset", "recovery_duration"}
    cols = [c for c in te.columns if c != "Id" and c not in drop]
    X = tr[cols].copy()
    nun = X.nunique(dropna=False)
    keep = [c for c in cols if nun[c] > 1]
    X = X[keep]
    for c in F.CAT_COLS:
        X[c] = X[c].astype("category")
    return X, tr


def rank_features(X, y, params, is_clf, seed=42):
    """Gain importance from a model fit on the training part only."""
    M = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor
    m = M(random_state=seed, n_jobs=-1, verbose=-1, **params)
    m.fit(X, y)
    return pd.Series(m.booster_.feature_importance("gain"), index=X.columns)


def sweep(X, y, params, is_clf, metric, stratify_on, label):
    """Rank ONCE per (seed, fold), then reuse that ranking for every k."""
    print("\n=== " + label + " ===", flush=True)
    ks = [k for k in KS if k <= X.shape[1]]
    oofs = {k: np.zeros((len(SEEDS), len(X))) for k in ks}
    M = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor

    for si, seed in enumerate(SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_i, va_i in skf.split(X, stratify_on):
            imp = rank_features(X.iloc[tr_i], y.iloc[tr_i], params, is_clf, seed)
            order = imp.sort_values(ascending=False).index.tolist()
            for k in ks:
                sel = order[:k]
                m = M(random_state=seed, n_jobs=-1, verbose=-1, **params)
                m.fit(X.iloc[tr_i][sel], y.iloc[tr_i])
                oofs[k][si, va_i] = (m.predict_proba(X.iloc[va_i][sel])[:, 1] if is_clf
                                     else m.predict(X.iloc[va_i][sel]))

    rows = []
    for k in ks:
        scores = [metric(y, oofs[k][si]) for si in range(len(SEEDS))]
        mu, sd = float(np.mean(scores)), float(np.std(scores))
        rows.append((k, mu, sd))
        print("  k=" + str(k).rjust(3) + " : " + format(mu, ".5f")
              + "  (+/-" + format(sd, ".5f") + ")", flush=True)
    best = (max(rows, key=lambda r: r[1]) if is_clf else min(rows, key=lambda r: r[1]))
    print("  -> best k=" + str(best[0]) + "  " + format(best[1], ".5f"), flush=True)
    return rows, best


def main():
    X, tr = load_xy()
    y = tr["injured_in_risk_window"]
    print("full matrix " + str(X.shape))

    out = {}

    clf_params = dict(objective="binary", n_estimators=600, learning_rate=0.02,
                      num_leaves=15, min_child_samples=30, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0)
    rows, best = sweep(X, y, clf_params, True, roc_auc_score, y,
                       "classifier: injured_in_risk_window (OOF AUC)")
    out["clf"] = {"rows": rows, "best_k": best[0], "best": best[1]}

    inj = (y == 1).to_numpy()
    Xi = X[inj].reset_index(drop=True)
    reg_params = dict(objective="l1", n_estimators=600, learning_rate=0.02,
                      num_leaves=15, min_child_samples=20, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0)
    for target in ["onset_day_offset", "recovery_duration"]:
        yi = tr.loc[inj, target].reset_index(drop=True)
        strat = pd.qcut(yi, 5, labels=False, duplicates="drop")
        rows, best = sweep(Xi, yi, reg_params, False, mean_absolute_error, strat,
                           "regressor: " + target + " (OOF MAE)")
        out[target] = {"rows": rows, "best_k": best[0], "best": best[1]}

    (ROOT / "reports" / "feature_sweep.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/feature_sweep.json")


if __name__ == "__main__":
    main()
