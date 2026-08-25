"""
AutoGluon (best_quality preset: bagged + multi-layer-stacked ensemble of
LightGBM/XGBoost/CatBoost/RF/ExtraTrees/NN/KNN, weighted at the top) vs our
hand-built ensemble. This is the standard SOTA AutoML reference point --
AutoGluon's best_quality preset tops most public tabular AutoML benchmarks.

Uses AutoGluon's own bagging (num_bag_folds) to get honest out-of-fold
predictions in one fit() call per head, rather than us re-implementing CV
around it. Same train/test split as everything else in this repo.
"""
from __future__ import annotations

import json
import sys
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_absolute_error
from autogluon.tabular import TabularPredictor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final import load_xy

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
AG_DIR = ROOT / "ag_models"

TIME_LIMIT = 300     # seconds per head
N_BAG_FOLDS = 5


def fit_head(train_df, test_df, label, problem_type, eval_metric, tag):
    path = AG_DIR / tag
    if path.exists():
        shutil.rmtree(path)
    predictor = TabularPredictor(label=label, problem_type=problem_type,
                                 eval_metric=eval_metric, path=str(path),
                                 verbosity=1)
    predictor.fit(train_df, presets="best_quality", time_limit=TIME_LIMIT,
                 num_bag_folds=N_BAG_FOLDS, num_bag_sets=1)
    return predictor


def main():
    X, Xt, tr, te = load_xy()
    y = tr["injured_in_risk_window"]
    results = {}

    print("=== AutoGluon classifier: injured_in_risk_window ===", flush=True)
    t0 = time.time()
    train_c = X.copy()
    train_c["target"] = y.values
    pred_c = fit_head(train_c, Xt, "target", "binary", "roc_auc", "clf")
    oof_proba = pred_c.predict_proba_oof()
    pos_col = [c for c in oof_proba.columns if str(c) in ("1", 1)][0]
    oof_c = oof_proba[pos_col].to_numpy()
    auc = roc_auc_score(y, oof_c)
    results["ag_auc"] = float(auc)
    test_proba_c = pred_c.predict_proba(Xt)[pos_col].to_numpy()
    print("  AUC " + format(auc, ".5f") + "  (" + format(time.time() - t0, ".0f") + "s)",
          flush=True)
    print(pred_c.leaderboard(silent=True).head(8).to_string(), flush=True)

    inj = (y == 1).to_numpy()
    Xi = X[inj].reset_index(drop=True)
    reg_test = {}
    for target, lo, hi in [("onset_day_offset", 1, 30), ("recovery_duration", 5, 20)]:
        yi = tr.loc[inj, target].reset_index(drop=True)
        print("\n=== AutoGluon regressor: " + target + " ===", flush=True)
        t0 = time.time()
        train_r = Xi.copy()
        train_r["target"] = yi.values
        pred_r = fit_head(train_r, Xt, "target", "regression", "mean_absolute_error",
                          "reg_" + target)
        oof_r = pred_r.predict_oof().to_numpy()
        mae = mean_absolute_error(yi, np.clip(oof_r, lo, hi))
        results["ag_mae_" + target] = float(mae)
        reg_test[target] = np.clip(pred_r.predict(Xt).to_numpy(), lo, hi)
        print("  MAE " + format(mae, ".4f") + "  (" + format(time.time() - t0, ".0f") + "s)",
              flush=True)
        print(pred_r.leaderboard(silent=True).head(8).to_string(), flush=True)

    M = json.loads((REPORTS / "metrics.json").read_text())
    tp_path = REPORTS / "bench_tabpfn.json"
    TP = json.loads(tp_path.read_text()) if tp_path.exists() else {}

    print("\n" + "=" * 68)
    print("  HEAD                       OURS      AutoGluon   TabPFN     best")
    print("=" * 68)
    a, b = M["auc_blend"], results["ag_auc"]
    c = TP.get("tabpfn_auc")
    row = {"ours": a, "AutoGluon": b, **({"TabPFN": c} if c is not None else {})}
    print("  injured (AUC, hi=better) " + format(a, ".4f") + "    " + format(b, ".4f")
          + "     " + (format(c, ".4f") if c is not None else "  -   ")
          + "    " + max(row, key=row.get))
    for t in ["onset_day_offset", "recovery_duration"]:
        a, b = M["mae_" + t], results["ag_mae_" + t]
        c = TP.get("tabpfn_mae_" + t)
        row = {"ours": a, "AutoGluon": b, **({"TabPFN": c} if c is not None else {})}
        print("  " + t.ljust(18) + " (MAE, lo=better) " + format(a, ".4f") + "    "
              + format(b, ".4f") + "     "
              + (format(c, ".4f") if c is not None else "  -   ")
              + "    " + min(row, key=row.get))
    print("=" * 68)

    (REPORTS / "bench_autogluon.json").write_text(json.dumps(results, indent=2))
    print("wrote reports/bench_autogluon.json")


if __name__ == "__main__":
    main()
