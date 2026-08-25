"""
Model training for PlayHack ML Track 2026 (IIT Guwahati).

Three targets, all predicted from the 30-day observation window only:
  injured_in_risk_window  binary   -> ROC-AUC (+ F1 at a tuned threshold)
  onset_day_offset        1..30    -> MAE, trained on injured athletes only
  recovery_duration       5..20    -> MAE, trained on injured athletes only

Ensemble: LightGBM + XGBoost + CatBoost, blended on out-of-fold predictions
with non-negative weights chosen to optimise the target metric. CV is repeated
over several seeds because n=3000 (1050 injured) makes single-split estimates
noisy.

The organisers never published a metric (see docs/problem_statement.md), so we
report AUC/F1/MAE and tune each head on its own natural metric.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_absolute_error, f1_score
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

SEEDS = [42, 202, 7]          # repeated CV: averages away fold-assignment noise
N_FOLDS = 5
GPU = True                    # falls back to CPU automatically if unavailable


# --------------------------------------------------------------- data prep ---
def load_xy():
    tr = F.get_features("train")
    te = F.get_features("test")
    labels = pd.read_csv(F.DATA / "train" / "train_labels.csv")

    tr = tr.merge(labels, left_on="Id", right_on="athlete_id", how="inner")
    target_cols = {"athlete_id", "injured_in_risk_window", "onset_day_offset",
                   "recovery_duration"}
    feat_cols = [c for c in te.columns if c != "Id" and c not in target_cols]

    X = tr[feat_cols].copy()
    X_test = te[feat_cols].copy()

    # drop constant columns (e.g. n_days_logged is 30 for everyone by construction)
    nunique = X.nunique(dropna=False)
    keep = [c for c in feat_cols if nunique[c] > 1]
    dropped = sorted(set(feat_cols) - set(keep))
    if dropped:
        print("dropped " + str(len(dropped)) + " constant cols: " + ", ".join(dropped[:8])
              + ("..." if len(dropped) > 8 else ""))
    X, X_test = X[keep], X_test[keep]

    for c in F.CAT_COLS:
        cats = pd.api.types.union_categoricals(
            [X[c].astype("category"), X_test[c].astype("category")]).categories
        X[c] = pd.Categorical(X[c], categories=cats)
        X_test[c] = pd.Categorical(X_test[c], categories=cats)

    return X, X_test, tr, te, keep


def to_numeric(df):
    """Ordinal-encode categoricals for XGBoost/CatBoost."""
    out = df.copy()
    for c in F.CAT_COLS:
        out[c] = out[c].cat.codes
    return out


# ------------------------------------------------------------- CV machinery ---
def cv_oof(fit_predict, X, y, X_test, stratify, seeds=SEEDS, n_folds=N_FOLDS):
    """Repeated K-fold. Returns (oof averaged over seeds, test averaged over all fits)."""
    oof = np.zeros((len(seeds), len(X)))
    test = np.zeros(len(X_test))
    n_fits = 0
    for si, seed in enumerate(seeds):
        if stratify:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            split = splitter.split(X, y)
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            split = splitter.split(X)
        for tr_idx, va_idx in split:
            p_va, p_te = fit_predict(X.iloc[tr_idx], y.iloc[tr_idx],
                                     X.iloc[va_idx], X_test, seed)
            oof[si, va_idx] = p_va
            test += p_te
            n_fits += 1
    return oof.mean(axis=0), test / n_fits


def blend_weights(oof_list, y, metric, maximize):
    """Non-negative weights summing to 1 that optimise `metric` on OOF preds."""
    P = np.column_stack(oof_list)
    k = P.shape[1]

    def obj(w):
        w = np.abs(w)
        s = w.sum()
        if s == 0:
            return 1e9
        val = metric(y, P @ (w / s))
        return -val if maximize else val

    best, best_val = np.ones(k) / k, obj(np.ones(k) / k)
    for start in [np.ones(k) / k] + [np.eye(k)[i] + 1e-3 for i in range(k)]:
        r = minimize(obj, start, method="Nelder-Mead",
                     options={"maxiter": 800, "xatol": 1e-4, "fatol": 1e-6})
        if r.fun < best_val:
            best_val, best = r.fun, np.abs(r.x)
    return best / best.sum(), (-best_val if maximize else best_val)


# ------------------------------------------------------------------ models ---
def lgb_clf(params):
    def f(Xtr, ytr, Xva, Xte, seed):
        m = lgb.LGBMClassifier(random_state=seed, verbose=-1, n_jobs=-1, **params)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    return f


def xgb_clf(params):
    def f(Xtr, ytr, Xva, Xte, seed):
        kw = dict(tree_method="hist", random_state=seed, n_jobs=-1,
                  eval_metric="auc", **params)
        if GPU:
            kw["device"] = "cuda"
        m = xgb.XGBClassifier(**kw)
        m.fit(to_numeric(Xtr), ytr, verbose=False)
        return (m.predict_proba(to_numeric(Xva))[:, 1],
                m.predict_proba(to_numeric(Xte))[:, 1])
    return f


def cat_clf(params):
    def f(Xtr, ytr, Xva, Xte, seed):
        m = CatBoostClassifier(random_seed=seed, verbose=0,
                               cat_features=F.CAT_COLS, **params)
        m.fit(Xtr.assign(**{c: Xtr[c].astype(str) for c in F.CAT_COLS}), ytr)
        pv = m.predict_proba(Xva.assign(**{c: Xva[c].astype(str) for c in F.CAT_COLS}))[:, 1]
        pt = m.predict_proba(Xte.assign(**{c: Xte[c].astype(str) for c in F.CAT_COLS}))[:, 1]
        return pv, pt
    return f


def lgb_reg(params):
    def f(Xtr, ytr, Xva, Xte, seed):
        m = lgb.LGBMRegressor(random_state=seed, verbose=-1, n_jobs=-1, **params)
        m.fit(Xtr, ytr)
        return m.predict(Xva), m.predict(Xte)
    return f


def xgb_reg(params):
    def f(Xtr, ytr, Xva, Xte, seed):
        kw = dict(tree_method="hist", random_state=seed, n_jobs=-1, **params)
        if GPU:
            kw["device"] = "cuda"
        m = xgb.XGBRegressor(**kw)
        m.fit(to_numeric(Xtr), ytr, verbose=False)
        return m.predict(to_numeric(Xva)), m.predict(to_numeric(Xte))
    return f


def cat_reg(params):
    def f(Xtr, ytr, Xva, Xte, seed):
        m = CatBoostRegressor(random_seed=seed, verbose=0,
                              cat_features=F.CAT_COLS, **params)
        m.fit(Xtr.assign(**{c: Xtr[c].astype(str) for c in F.CAT_COLS}), ytr)
        return (m.predict(Xva.assign(**{c: Xva[c].astype(str) for c in F.CAT_COLS})),
                m.predict(Xte.assign(**{c: Xte[c].astype(str) for c in F.CAT_COLS})))
    return f


# ----------------------------------------------------------------- helpers ---
def best_f1_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y, (p >= t).astype(int)) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def main():
    X, X_test, tr, te, feat_cols = load_xy()
    y_cls = tr["injured_in_risk_window"]
    print("features: " + str(X.shape) + "  test: " + str(X_test.shape)
          + "  positives: " + str(int(y_cls.sum())))

    metrics = {"n_features": int(X.shape[1])}

    # ---------------------------------------------------- classification ----
    clf_specs = {
        "lgb": lgb_clf(dict(objective="binary", n_estimators=700, learning_rate=0.02,
                            num_leaves=31, min_child_samples=25, subsample=0.8,
                            subsample_freq=1, colsample_bytree=0.6, reg_lambda=3.0,
                            reg_alpha=0.5)),
        "xgb": xgb_clf(dict(n_estimators=700, learning_rate=0.02, max_depth=5,
                            min_child_weight=5, subsample=0.8, colsample_bytree=0.6,
                            reg_lambda=3.0, reg_alpha=0.5)),
        "cat": cat_clf(dict(iterations=800, learning_rate=0.03, depth=6,
                            l2_leaf_reg=6.0)),
    }
    oof_cls, test_cls = {}, {}
    for name, fn in clf_specs.items():
        o, t = cv_oof(fn, X, y_cls, X_test, stratify=True)
        oof_cls[name], test_cls[name] = o, t
        auc = roc_auc_score(y_cls, o)
        metrics["auc_" + name] = auc
        print("  clf " + name + ": OOF AUC " + format(auc, ".5f"))

    names = list(oof_cls)
    w, blend_auc = blend_weights([oof_cls[n] for n in names], y_cls,
                                 roc_auc_score, maximize=True)
    oof_blend = np.column_stack([oof_cls[n] for n in names]) @ w
    test_blend = np.column_stack([test_cls[n] for n in names]) @ w
    metrics["auc_blend"] = float(blend_auc)
    metrics["blend_weights"] = {n: float(x) for n, x in zip(names, w)}
    print("  clf BLEND: OOF AUC " + format(blend_auc, ".5f")
          + "  weights " + str(metrics["blend_weights"]))

    thr, f1 = best_f1_threshold(y_cls, oof_blend)
    metrics["best_threshold"] = thr
    metrics["f1_at_threshold"] = f1
    metrics["f1_at_0.5"] = float(f1_score(y_cls, (oof_blend >= 0.5).astype(int)))
    print("  clf F1: " + format(f1, ".5f") + " @ thr " + format(thr, ".3f")
          + "   (F1@0.5 = " + format(metrics["f1_at_0.5"], ".5f") + ")")

    # -------------------------------------------------------- regression ----
    inj = (y_cls == 1).to_numpy()
    X_inj = X[inj].reset_index(drop=True)
    reg_test = {}
    for target, lo, hi in [("onset_day_offset", 1, 30), ("recovery_duration", 5, 20)]:
        y_reg = tr.loc[inj, target].reset_index(drop=True)
        specs = {
            # L1 objectives: the natural loss when the metric is MAE
            "lgb": lgb_reg(dict(objective="l1", n_estimators=700, learning_rate=0.02,
                                num_leaves=15, min_child_samples=20, subsample=0.8,
                                subsample_freq=1, colsample_bytree=0.6, reg_lambda=3.0)),
            "xgb": xgb_reg(dict(objective="reg:absoluteerror", n_estimators=700,
                                learning_rate=0.02, max_depth=4, min_child_weight=5,
                                subsample=0.8, colsample_bytree=0.6, reg_lambda=3.0)),
            "cat": cat_reg(dict(loss_function="MAE", iterations=800,
                                learning_rate=0.03, depth=5, l2_leaf_reg=6.0)),
        }
        oof_r, test_r = {}, {}
        for name, fn in specs.items():
            o, t = cv_oof(fn, X_inj, y_reg, X_test, stratify=False)
            oof_r[name], test_r[name] = o, t
            mae = mean_absolute_error(y_reg, o)
            metrics["mae_" + target + "_" + name] = mae
            print("  reg " + target + " " + name + ": OOF MAE " + format(mae, ".4f"))

        rn = list(oof_r)
        wr, blend_mae = blend_weights([oof_r[n] for n in rn], y_reg,
                                      mean_absolute_error, maximize=False)
        metrics["mae_" + target + "_blend"] = float(blend_mae)
        metrics["blend_weights_" + target] = {n: float(x) for n, x in zip(rn, wr)}
        print("  reg " + target + " BLEND: OOF MAE " + format(blend_mae, ".4f")
              + "  weights " + str(metrics["blend_weights_" + target]))
        reg_test[target] = np.clip(
            np.column_stack([test_r[n] for n in rn]) @ wr, lo, hi)

    # -------------------------------------------------------- submission ----
    pred_injured = (test_blend >= thr).astype(int)
    submission = pd.DataFrame({
        "athlete_id": te["Id"].values,
        "injured_in_risk_window": pred_injured,
        # regressor values for every athlete: ground truth for onset/recovery only
        # exists for truly-injured athletes, so MAE can only be scored on those
        # rows -- filling a constant for our predicted-negatives would throw away
        # a real prediction on any athlete we misclassify.
        "onset_day_offset": np.round(reg_test["onset_day_offset"]).astype(int),
        "recovery_duration": np.round(reg_test["recovery_duration"]).astype(int),
    })
    submission.to_csv(ROOT / "submission.csv", index=False)
    metrics["test_positive_rate"] = float(pred_injured.mean())
    metrics["train_positive_rate"] = float(y_cls.mean())

    np.save(REPORTS / "oof_blend.npy", oof_blend)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items()
                      if not k.startswith("blend_weights")}, indent=2))
    return metrics


if __name__ == "__main__":
    main()
