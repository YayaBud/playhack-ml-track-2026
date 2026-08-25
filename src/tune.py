"""
Optuna hyper-parameter search for the three PlayHack heads.

Small data (3000 athletes, 1050 injured) means a single 5-fold estimate moves
around by ~0.005 AUC on fold assignment alone, which is the same size as the
gains being searched for. Every trial is therefore scored with 2-seed repeated
5-fold CV, and the winning configs are written to models/best_params.json for
train.py to pick up.

Usage:  python tune.py [n_trials]
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T
import features as F

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "best_params.json"
TUNE_SEEDS = [42, 202]


def _cv_auc(fit_predict, X, y, X_test):
    oof, _ = T.cv_oof(fit_predict, X, y, X_test, stratify=True, seeds=TUNE_SEEDS)
    return roc_auc_score(y, oof)


def _cv_mae(fit_predict, X, y, X_test):
    oof, _ = T.cv_oof(fit_predict, X, y, X_test, stratify=False, seeds=TUNE_SEEDS)
    return mean_absolute_error(y, oof)


def tune_classifier(X, y, X_test, n_trials):
    best = {}

    def lgb_obj(t):
        p = dict(objective="binary",
                 n_estimators=t.suggest_int("n_estimators", 300, 1500, step=100),
                 learning_rate=t.suggest_float("learning_rate", 0.005, 0.06, log=True),
                 num_leaves=t.suggest_int("num_leaves", 7, 63),
                 min_child_samples=t.suggest_int("min_child_samples", 10, 80),
                 subsample=t.suggest_float("subsample", 0.5, 1.0),
                 subsample_freq=1,
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
                 reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10.0, log=True))
        return _cv_auc(T.lgb_clf(p), X, y, X_test)

    def xgb_obj(t):
        p = dict(n_estimators=t.suggest_int("n_estimators", 300, 1500, step=100),
                 learning_rate=t.suggest_float("learning_rate", 0.005, 0.06, log=True),
                 max_depth=t.suggest_int("max_depth", 3, 9),
                 min_child_weight=t.suggest_int("min_child_weight", 1, 20),
                 subsample=t.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
                 reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10.0, log=True))
        return _cv_auc(T.xgb_clf(p), X, y, X_test)

    def cat_obj(t):
        p = dict(iterations=t.suggest_int("iterations", 300, 1500, step=100),
                 learning_rate=t.suggest_float("learning_rate", 0.01, 0.10, log=True),
                 depth=t.suggest_int("depth", 4, 8),
                 l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True))
        return _cv_auc(T.cat_clf(p), X, y, X_test)

    for name, obj in [("lgb", lgb_obj), ("xgb", xgb_obj), ("cat", cat_obj)]:
        st = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=0))
        st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        best[name] = st.best_params
        print("  clf " + name + " best AUC " + format(st.best_value, ".5f"))
    return best


def tune_regressor(X_inj, y_reg, X_test, target, n_trials):
    best = {}

    def lgb_obj(t):
        p = dict(objective="l1",
                 n_estimators=t.suggest_int("n_estimators", 300, 1500, step=100),
                 learning_rate=t.suggest_float("learning_rate", 0.005, 0.06, log=True),
                 num_leaves=t.suggest_int("num_leaves", 7, 63),
                 min_child_samples=t.suggest_int("min_child_samples", 5, 60),
                 subsample=t.suggest_float("subsample", 0.5, 1.0),
                 subsample_freq=1,
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-2, 30.0, log=True))
        return _cv_mae(T.lgb_reg(p), X_inj, y_reg, X_test)

    def xgb_obj(t):
        p = dict(objective="reg:absoluteerror",
                 n_estimators=t.suggest_int("n_estimators", 300, 1500, step=100),
                 learning_rate=t.suggest_float("learning_rate", 0.005, 0.06, log=True),
                 max_depth=t.suggest_int("max_depth", 3, 8),
                 min_child_weight=t.suggest_int("min_child_weight", 1, 20),
                 subsample=t.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-2, 30.0, log=True))
        return _cv_mae(T.xgb_reg(p), X_inj, y_reg, X_test)

    def cat_obj(t):
        p = dict(loss_function="MAE",
                 iterations=t.suggest_int("iterations", 300, 1500, step=100),
                 learning_rate=t.suggest_float("learning_rate", 0.01, 0.10, log=True),
                 depth=t.suggest_int("depth", 4, 8),
                 l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True))
        return _cv_mae(T.cat_reg(p), X_inj, y_reg, X_test)

    for name, obj in [("lgb", lgb_obj), ("xgb", xgb_obj), ("cat", cat_obj)]:
        st = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=0))
        st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        best[name] = st.best_params
        print("  reg " + target + " " + name + " best MAE " + format(st.best_value, ".4f"))
    return best


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    X, X_test, tr, te, feat_cols = T.load_xy()
    y = tr["injured_in_risk_window"]
    print("tuning on " + str(X.shape) + " with " + str(n_trials) + " trials/model")

    out = {"classifier": tune_classifier(X, y, X_test, n_trials)}

    inj = (y == 1).to_numpy()
    X_inj = X[inj].reset_index(drop=True)
    for target in ["onset_day_offset", "recovery_duration"]:
        y_reg = tr.loc[inj, target].reset_index(drop=True)
        out[target] = tune_regressor(X_inj, y_reg, X_test, target, n_trials)

    OUT.write_text(json.dumps(out, indent=2))
    print("wrote " + str(OUT))


if __name__ == "__main__":
    main()
