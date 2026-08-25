"""
Survival (AFT) formulation of the PlayHack injury task.

THE IDEA
--------
`injured_in_risk_window` and `onset_day_offset` are not two problems. They are
one latent time-to-injury T, coarsened:

    injured = 1(T <= 30),      onset = T given T <= 30

Modelling the binary label alone throws away two things:
  1. the 1950 healthy athletes are RIGHT-CENSORED observations (T > 30), not
     just negatives -- and the onset regressor currently never sees them;
  2. for the injured, the exact day carries far more information than the bit.

An accelerated-failure-time model fits log T on all 3000 athletes with proper
censoring, then both targets fall out of one fitted distribution:

    P(injured) = P(T <= 30) = Phi( (log 30 - mu) / sigma )
    onset      = median of T truncated to (0, 30]

Run directly to compare AFT against the plain classifier/regressor split.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F

warnings.filterwarnings("ignore")
SEED = 42
N_FOLDS = 5
HORIZON = 30.0


def load_xy():
    tr = F.get_features("train")
    te = F.get_features("test")
    lab = pd.read_csv(F.DATA / "train" / "train_labels.csv")
    tr = tr.merge(lab, left_on="Id", right_on="athlete_id")
    drop = {"athlete_id", "injured_in_risk_window", "onset_day_offset",
            "recovery_duration"}
    cols = [c for c in te.columns if c != "Id" and c not in drop]
    X = tr[cols].copy()
    Xt = te[cols].copy()
    nun = X.nunique(dropna=False)
    keep = [c for c in cols if nun[c] > 1]
    X, Xt = X[keep], Xt[keep]
    for c in F.CAT_COLS:
        X[c] = X[c].astype("category").cat.codes
        Xt[c] = Xt[c].astype("category").cat.codes
    return X, Xt, tr, te


def aft_bounds(tr):
    """Exact label for the injured, right-censored at the 30-day horizon for the rest."""
    injured = tr["injured_in_risk_window"].to_numpy() == 1
    onset = tr["onset_day_offset"].to_numpy(dtype=float)
    lo = np.where(injured, onset, HORIZON)
    hi = np.where(injured, onset, np.inf)
    return lo, hi


def fit_aft(Xtr, lo, hi, Xva, Xte, params, seed):
    dtr = xgb.DMatrix(Xtr, missing=np.nan)
    dtr.set_float_info("label_lower_bound", lo)
    dtr.set_float_info("label_upper_bound", hi)
    p = dict(objective="survival:aft", eval_metric="aft-nloglik",
             aft_loss_distribution="normal", tree_method="hist",
             seed=seed, **params)
    n = p.pop("num_boost_round", 600)
    bst = xgb.train(p, dtr, num_boost_round=n)
    mu_va = np.log(np.clip(bst.predict(xgb.DMatrix(Xva, missing=np.nan)), 1e-6, None))
    mu_te = np.log(np.clip(bst.predict(xgb.DMatrix(Xte, missing=np.nan)), 1e-6, None))
    return mu_va, mu_te


def decode(mu, sigma):
    """Turn AFT log-time into P(injured) and the truncated-median onset."""
    z = (np.log(HORIZON) - mu) / sigma
    p_inj = norm.cdf(z)
    # median of T | T <= 30 under lognormal(mu, sigma)
    q = norm.cdf(z) * 0.5
    onset = np.exp(mu + sigma * norm.ppf(np.clip(q, 1e-9, 1 - 1e-9)))
    return p_inj, np.clip(onset, 1, 30)


def main():
    X, Xt, tr, te = load_xy()
    y = tr["injured_in_risk_window"]
    lo, hi = aft_bounds(tr)
    inj = y.to_numpy() == 1
    print("X " + str(X.shape) + "  injured " + str(int(inj.sum()))
          + "  censored " + str(int((~inj).sum())))

    sigmas = [0.35, 0.5, 0.7, 0.9, 1.2]
    params_base = dict(learning_rate=0.03, max_depth=4, min_child_weight=8,
                       subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0,
                       num_boost_round=700)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    print("\n=== AFT: sigma sweep (5-fold OOF) ===")
    best = None
    for s in sigmas:
        mu_oof = np.zeros(len(X))
        for tr_i, va_i in folds:
            p = dict(params_base, aft_loss_distribution_scale=s)
            mu_va, _ = fit_aft(X.iloc[tr_i], lo[tr_i], hi[tr_i], X.iloc[va_i], Xt, p, SEED)
            mu_oof[va_i] = mu_va
        p_inj, onset = decode(mu_oof, s)
        auc = roc_auc_score(y, p_inj)
        mae = mean_absolute_error(tr.loc[inj, "onset_day_offset"], onset[inj])
        # AUC is invariant to sigma (monotone in mu) but MAE is not
        print("  sigma " + format(s, ".2f") + " : AUC " + format(auc, ".5f")
              + "   onset MAE " + format(mae, ".4f"))
        if best is None or mae < best[1]:
            best = (s, mae, auc, mu_oof)

    s, mae, auc, mu_oof = best
    print("  -> best sigma " + format(s, ".2f") + "  (onset MAE "
          + format(mae, ".4f") + ", AUC " + format(auc, ".5f") + ")")

    # ---- reference: the conventional two-model split -----------------------
    print("\n=== reference: separate classifier / regressor ===")
    clf_p = dict(objective="binary", n_estimators=700, learning_rate=0.02,
                 num_leaves=31, min_child_samples=25, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.6, reg_lambda=3.0, verbose=-1)
    oof_clf = np.zeros(len(X))
    for tr_i, va_i in folds:
        m = lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, **clf_p)
        m.fit(X.iloc[tr_i], y.iloc[tr_i])
        oof_clf[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
    auc_clf = roc_auc_score(y, oof_clf)
    print("  LGBM classifier                 : AUC " + format(auc_clf, ".5f"))

    Xi = X[inj].reset_index(drop=True)
    yi = tr.loc[inj, "onset_day_offset"].reset_index(drop=True)
    reg_p = dict(objective="l1", n_estimators=700, learning_rate=0.02,
                 num_leaves=15, min_child_samples=20, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.6, reg_lambda=3.0, verbose=-1)
    oof_reg = np.zeros(len(Xi))
    for tr_i, va_i in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
            Xi, pd.qcut(yi, 5, labels=False, duplicates="drop")):
        m = lgb.LGBMRegressor(random_state=SEED, n_jobs=-1, **reg_p)
        m.fit(Xi.iloc[tr_i], yi.iloc[tr_i])
        oof_reg[va_i] = m.predict(Xi.iloc[va_i])
    mae_reg = mean_absolute_error(yi, np.clip(oof_reg, 1, 30))
    print("  LGBM L1 regressor (injured only): onset MAE " + format(mae_reg, ".4f"))

    print("\n=== verdict ===")
    print("  AUC       AFT " + format(auc, ".5f") + "  vs  classifier "
          + format(auc_clf, ".5f") + "   delta " + format(auc - auc_clf, "+.5f"))
    print("  onset MAE AFT " + format(mae, ".4f") + "  vs  regressor  "
          + format(mae_reg, ".4f") + "   delta " + format(mae - mae_reg, "+.4f"))

    # blending the two ranking signals usually beats either
    from scipy.stats import rankdata
    bl = 0.5 * rankdata(mu_oof) / len(mu_oof) + 0.5 * (1 - rankdata(oof_clf) / len(oof_clf))
    print("  AUC rank-blend(AFT, clf)        : " + format(roc_auc_score(y, -bl), ".5f"))
    np.save(Path(__file__).resolve().parents[1] / "reports" / "aft_mu_oof.npy", mu_oof)


if __name__ == "__main__":
    main()
