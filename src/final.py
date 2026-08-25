"""
Final PlayHack pipeline: ensemble + Bayes-ceiling comparison.

Pipeline per head
-----------------
1. Feature selection INSIDE each CV fold (gain-ranked on the training part
   only, so the held-out fold never influences which columns are kept). The
   223-column matrix is heavily redundant -- the ACWR family is r~0.99 within
   itself -- and the full matrix scores below a single raw `hr_acwr`.
2. LightGBM + XGBoost + CatBoost, repeated over several seeds.
3. Non-negative blend weights fitted on the out-of-fold matrix.

Ceiling
-------
The label is a coin flip with probability p(x): identical athletes can land on
opposite labels, so no model reaches AUC 1.0. Given the true p, the expected
AUC of the Bayes ranking is closed-form (see oracle.expected_auc). We estimate
p by cross-fitted isotonic calibration of the blended OOF score and evaluate
that formula -- giving a ceiling to measure the model against rather than an
unreachable 1.0.

Under MAE the Bayes act is the conditional MEDIAN, so the ceiling for the two
day-count heads is the mean absolute deviation of the conditional distribution
around its median, estimated within bins of the model's own prediction.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_absolute_error, f1_score
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from oracle import expected_auc

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

SEEDS = [42, 202, 7]
N_FOLDS = 5
GPU = True

# chosen by src/select_features.py (reports/feature_sweep.json)
K_CLF = 35
K_ONSET = 35
K_RECOVERY = 20


# --------------------------------------------------------------------- data ---
def load_xy():
    tr = F.get_features("train")
    te = F.get_features("test")
    lab = pd.read_csv(F.DATA / "train" / "train_labels.csv")
    tr = tr.merge(lab, left_on="Id", right_on="athlete_id")
    drop = {"athlete_id", "injured_in_risk_window", "onset_day_offset",
            "recovery_duration"}
    cols = [c for c in te.columns if c != "Id" and c not in drop]
    X, Xt = tr[cols].copy(), te[cols].copy()
    nun = X.nunique(dropna=False)
    keep = [c for c in cols if nun[c] > 1]
    X, Xt = X[keep], Xt[keep]
    for c in F.CAT_COLS:
        cats = pd.api.types.union_categoricals(
            [X[c].astype("category"), Xt[c].astype("category")]).categories
        X[c] = pd.Categorical(X[c], categories=cats)
        Xt[c] = pd.Categorical(Xt[c], categories=cats)
    return X, Xt, tr, te


def num(df):
    out = df.copy()
    for c in F.CAT_COLS:
        if c in out.columns:
            out[c] = out[c].cat.codes
    return out


def rank_cols(X, y, is_clf, seed):
    M = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor
    p = (dict(objective="binary") if is_clf else dict(objective="l1"))
    m = M(n_estimators=400, learning_rate=0.03, num_leaves=15,
          min_child_samples=25, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.7, reg_lambda=5.0, random_state=seed,
          n_jobs=-1, verbose=-1, **p)
    m.fit(X, y)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
    return imp.sort_values(ascending=False).index.tolist()


# ------------------------------------------------------------------ models ---
def make_clf(kind, seed):
    if kind == "lgb":
        return lgb.LGBMClassifier(objective="binary", n_estimators=600,
                                  learning_rate=0.02, num_leaves=15,
                                  min_child_samples=30, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.7,
                                  reg_lambda=5.0, random_state=seed,
                                  n_jobs=-1, verbose=-1)
    if kind == "xgb":
        kw = dict(n_estimators=600, learning_rate=0.02, max_depth=4,
                  min_child_weight=8, subsample=0.8, colsample_bytree=0.7,
                  reg_lambda=5.0, tree_method="hist", random_state=seed,
                  n_jobs=-1, eval_metric="auc")
        if GPU:
            kw["device"] = "cuda"
        return xgb.XGBClassifier(**kw)
    return CatBoostClassifier(iterations=700, learning_rate=0.03, depth=5,
                              l2_leaf_reg=8.0, random_seed=seed, verbose=0)


def make_reg(kind, seed):
    if kind == "lgb":
        return lgb.LGBMRegressor(objective="l1", n_estimators=600,
                                 learning_rate=0.02, num_leaves=15,
                                 min_child_samples=20, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.7,
                                 reg_lambda=5.0, random_state=seed,
                                 n_jobs=-1, verbose=-1)
    if kind == "xgb":
        kw = dict(objective="reg:absoluteerror", n_estimators=600,
                  learning_rate=0.02, max_depth=4, min_child_weight=8,
                  subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
                  tree_method="hist", random_state=seed, n_jobs=-1)
        if GPU:
            kw["device"] = "cuda"
        return xgb.XGBRegressor(**kw)
    return CatBoostRegressor(loss_function="MAE", iterations=700,
                             learning_rate=0.03, depth=5, l2_leaf_reg=8.0,
                             random_seed=seed, verbose=0)


def run_head(X, y, Xt, k, is_clf, strat, kinds=("lgb", "xgb", "cat")):
    """Repeated CV with in-fold feature selection. Returns OOF and test preds."""
    oof = {kd: np.zeros((len(SEEDS), len(X))) for kd in kinds}
    test = {kd: np.zeros(len(Xt)) for kd in kinds}
    n_fits = 0
    for si, seed in enumerate(SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_i, va_i in skf.split(X, strat):
            order = rank_cols(X.iloc[tr_i], y.iloc[tr_i], is_clf, seed)
            sel = order[:k]
            for kd in kinds:
                m = make_clf(kd, seed) if is_clf else make_reg(kd, seed)
                if kd == "cat":
                    cats = [c for c in F.CAT_COLS if c in sel]
                    A = X.iloc[tr_i][sel].copy()
                    B = X.iloc[va_i][sel].copy()
                    C = Xt[sel].copy()
                    for c in cats:
                        A[c], B[c], C[c] = A[c].astype(str), B[c].astype(str), C[c].astype(str)
                    m.set_params(cat_features=cats)
                    m.fit(A, y.iloc[tr_i])
                elif kd == "xgb":
                    A, B, C = num(X.iloc[tr_i][sel]), num(X.iloc[va_i][sel]), num(Xt[sel])
                    m.fit(A, y.iloc[tr_i], verbose=False)
                else:
                    A, B, C = X.iloc[tr_i][sel], X.iloc[va_i][sel], Xt[sel]
                    m.fit(A, y.iloc[tr_i])
                if is_clf:
                    oof[kd][si, va_i] = m.predict_proba(B)[:, 1]
                    test[kd] += m.predict_proba(C)[:, 1]
                else:
                    oof[kd][si, va_i] = m.predict(B)
                    test[kd] += m.predict(C)
            n_fits += 1
    n_per = n_fits
    return ({kd: oof[kd].mean(axis=0) for kd in kinds},
            {kd: test[kd] / n_per for kd in kinds})


def blend(oof_map, y, metric, maximize):
    names = list(oof_map)
    P = np.column_stack([oof_map[n] for n in names])

    def obj(w):
        w = np.abs(w)
        s = w.sum()
        if s == 0:
            return 1e9
        v = metric(y, P @ (w / s))
        return -v if maximize else v

    best, bv = np.ones(len(names)) / len(names), obj(np.ones(len(names)) / len(names))
    for st in [np.ones(len(names)) / len(names)] + [np.eye(len(names))[i] + 1e-3
                                                    for i in range(len(names))]:
        r = minimize(obj, st, method="Nelder-Mead",
                     options={"maxiter": 600, "fatol": 1e-7})
        if r.fun < bv:
            bv, best = r.fun, np.abs(r.x)
    w = best / best.sum()
    return dict(zip(names, w)), (-bv if maximize else bv)


def calibrated_p(score, y):
    """Cross-fitted isotonic calibration of an OOF score into probabilities."""
    p = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEEDS[0])
    for tr_i, va_i in skf.split(score.reshape(-1, 1), y):
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(score[tr_i], y.iloc[tr_i].to_numpy())
        p[va_i] = iso.predict(score[va_i])
    return np.clip(p, 1e-6, 1 - 1e-6)


def conditional_mad_ceiling(pred, y, n_bins=25):
    """Ceiling for an MAE head: mean within-bin absolute deviation from the median.

    Bins are formed on the model's own prediction, so each bin approximates a
    set of athletes the model considers identical; the spread left inside a bin
    is irreducible under this feature set.
    """
    q = pd.qcut(pd.Series(pred), n_bins, labels=False, duplicates="drop")
    s = pd.Series(np.asarray(y), index=q.index)
    return float(s.groupby(q).transform(lambda v: (v - v.median()).abs().mean()).mean())


def main():
    X, Xt, tr, te = load_xy()
    y = tr["injured_in_risk_window"]
    print("matrix " + str(X.shape) + "  test " + str(Xt.shape)
          + "  positives " + str(int(y.sum())), flush=True)
    M = {}

    # ------------------------------------------------- classification -------
    print("\n=== injured_in_risk_window (k=" + str(K_CLF) + ") ===", flush=True)
    oof_c, test_c = run_head(X, y, Xt, K_CLF, True, y)
    for kd, v in oof_c.items():
        M["auc_" + kd] = float(roc_auc_score(y, v))
        print("  " + kd + " : AUC " + format(M["auc_" + kd], ".5f"), flush=True)
    wc, auc_b = blend(oof_c, y, roc_auc_score, True)
    oof_blend = np.column_stack([oof_c[k] for k in wc]) @ np.array(list(wc.values()))
    test_blend = np.column_stack([test_c[k] for k in wc]) @ np.array(list(wc.values()))
    M["auc_blend"] = float(auc_b)
    M["blend_weights_clf"] = {k: float(v) for k, v in wc.items()}
    print("  BLEND: AUC " + format(auc_b, ".5f") + "  " + str(M["blend_weights_clf"]), flush=True)

    p_cal = calibrated_p(oof_blend, y)
    M["oracle_auc"] = float(expected_auc(p_cal))
    grid = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y, (oof_blend >= t).astype(int)) for t in grid]
    M["best_threshold"] = float(grid[int(np.argmax(f1s))])
    M["f1"] = float(max(f1s))
    of1 = []
    for t in grid:
        pr = p_cal >= t
        tp = (p_cal * pr).sum(); fp = ((1 - p_cal) * pr).sum(); fn = (p_cal * ~pr).sum()
        of1.append(2 * tp / max(2 * tp + fp + fn, 1e-9))
    M["oracle_f1"] = float(max(of1))
    print("  CEILING: AUC " + format(M["oracle_auc"], ".5f")
          + "   F1 " + format(M["oracle_f1"], ".5f"), flush=True)
    print("  MODEL  : AUC " + format(auc_b, ".5f")
          + "   F1 " + format(M["f1"], ".5f")
          + " @thr " + format(M["best_threshold"], ".3f"), flush=True)

    # ------------------------------------------------------ regression ------
    inj = (y == 1).to_numpy()
    Xi = X[inj].reset_index(drop=True)
    reg_test = {}
    for target, k, lo, hi in [("onset_day_offset", K_ONSET, 1, 30),
                              ("recovery_duration", K_RECOVERY, 5, 20)]:
        yi = tr.loc[inj, target].reset_index(drop=True)
        strat = pd.qcut(yi, 5, labels=False, duplicates="drop")
        print("\n=== " + target + " (k=" + str(k) + ", n=" + str(len(yi)) + ") ===", flush=True)
        oof_r, test_r = run_head(Xi, yi, Xt, k, False, strat)
        for kd, v in oof_r.items():
            M["mae_" + target + "_" + kd] = float(mean_absolute_error(yi, np.clip(v, lo, hi)))
            print("  " + kd + " : MAE " + format(M["mae_" + target + "_" + kd], ".4f"), flush=True)
        wr, mae_b = blend(oof_r, yi, mean_absolute_error, False)
        ob = np.column_stack([oof_r[q] for q in wr]) @ np.array(list(wr.values()))
        tb = np.column_stack([test_r[q] for q in wr]) @ np.array(list(wr.values()))
        M["mae_" + target] = float(mean_absolute_error(yi, np.clip(ob, lo, hi)))
        M["blend_weights_" + target] = {q: float(v) for q, v in wr.items()}
        M["constant_mae_" + target] = float(
            mean_absolute_error(yi, np.full(len(yi), yi.median())))
        M["oracle_mae_" + target] = conditional_mad_ceiling(ob, yi)
        print("  BLEND  : MAE " + format(M["mae_" + target], ".4f")
              + "   " + str(M["blend_weights_" + target]), flush=True)
        print("  CEILING: MAE " + format(M["oracle_mae_" + target], ".4f")
              + "   (constant-median baseline "
              + format(M["constant_mae_" + target], ".4f") + ")", flush=True)
        reg_test[target] = np.clip(tb, lo, hi)
        # cache so threshold/rounding choices can be revisited without refitting
        np.save(REPORTS / ("oof_" + target + ".npy"), ob)
        np.save(REPORTS / ("test_" + target + ".npy"), tb)

    # ------------------------------------------------------ submission ------
    # An OOF-tuned ABSOLUTE threshold must not be applied to the test scores.
    # Each OOF row is predicted by 3 models (one per seed) while each test row
    # averages all 45 fold-models, and averaging compresses the spread -- so the
    # same cut-off fires far less often on test. Left uncorrected this shipped a
    # 25.5% positive rate when the test ACWR distribution actually implies ~38%
    # (test is the higher-risk split: 22.0% of athletes sit above the overload
    # threshold vs 17.5% in train).
    #
    # Fix: map both score sets onto the probability scale with an isotonic
    # calibrator fitted on OOF, then choose the threshold that maximises the
    # EXPECTED F1 of the test set under its own calibrated probabilities. This
    # removes the averaging artefact while preserving the genuine risk shift.
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(oof_blend, y.to_numpy())
    p_test = np.clip(iso.predict(test_blend), 1e-6, 1 - 1e-6)

    best_t, best_ef1 = 0.5, -1.0
    for t in grid:
        pr = p_test >= t
        tp = (p_test * pr).sum()
        fp = ((1 - p_test) * pr).sum()
        fn = (p_test * ~pr).sum()
        ef1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if ef1 > best_ef1:
            best_ef1, best_t = ef1, t
    pred = (p_test >= best_t).astype(int)
    M["test_threshold_calibrated"] = float(best_t)
    M["test_expected_f1"] = float(best_ef1)
    M["test_expected_positive_rate"] = float(p_test.mean())
    M["naive_threshold_positive_rate"] = float((test_blend >= M["best_threshold"]).mean())
    print("\n  threshold: OOF-tuned raw " + format(M["best_threshold"], ".3f")
          + " would fire on " + format(M["naive_threshold_positive_rate"], ".4f")
          + " of test; calibrated cut " + format(best_t, ".3f")
          + " fires on " + format(pred.mean(), ".4f")
          + " (test E[rate] " + format(p_test.mean(), ".4f") + ")", flush=True)
    np.save(REPORTS / "test_blend.npy", test_blend)
    np.save(REPORTS / "p_test.npy", p_test)
    pd.DataFrame({
        "athlete_id": te["Id"].values,
        "injured_in_risk_window": pred,
        "onset_day_offset": np.round(reg_test["onset_day_offset"]).astype(int),
        "recovery_duration": np.round(reg_test["recovery_duration"]).astype(int),
    }).to_csv(ROOT / "submission.csv", index=False)
    M["test_positive_rate"] = float(pred.mean())
    M["train_positive_rate"] = float(y.mean())

    (REPORTS / "metrics.json").write_text(json.dumps(M, indent=2))
    np.save(REPORTS / "oof_blend.npy", oof_blend)

    print("\n" + "=" * 62)
    print("  HEAD                      MODEL     CEILING   BASELINE   % of gap")
    print("=" * 62)
    base_auc = 0.5
    got = (M["auc_blend"] - base_auc) / max(M["oracle_auc"] - base_auc, 1e-9) * 100
    print("  injured (AUC)            " + format(M["auc_blend"], ".4f") + "    "
          + format(M["oracle_auc"], ".4f") + "    " + format(base_auc, ".4f")
          + "     " + format(got, "5.1f") + "%")
    for t in ["onset_day_offset", "recovery_duration"]:
        b, c, m = M["constant_mae_" + t], M["oracle_mae_" + t], M["mae_" + t]
        got = (b - m) / max(b - c, 1e-9) * 100
        print("  " + t.replace("_", " ")[:22].ljust(23) + " " + format(m, ".4f")
              + "    " + format(c, ".4f") + "    " + format(b, ".4f")
              + "     " + format(got, "5.1f") + "%")
    print("=" * 62)
    print("wrote submission.csv + reports/metrics.json")
    return M


if __name__ == "__main__":
    main()
