"""
TabPFN v2 vs our hand-built ensemble.

TabPFN is a pretrained transformer that does in-context learning for tabular
data -- no per-dataset training, just a forward pass -- and is specifically
positioned as SOTA for exactly this data regime: <=10k rows, <=500 features,
few classes. Our data (3000 rows / 1050 injured, 206 features) is squarely in
its target zone, so it's the right "pull a SOTA model from the web" baseline
to check ourselves against.

Same CV protocol as src/final.py (repeated stratified 5-fold, same seeds) so
the comparison is apples-to-apples against reports/metrics.json.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_absolute_error

from tabpfn import TabPFNClassifier, TabPFNRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final import load_xy, num, SEEDS, N_FOLDS

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def run_clf(X, y, seeds=SEEDS, n_folds=N_FOLDS):
    oof = np.zeros((len(seeds), len(X)))
    for si, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr_i, va_i in skf.split(X, y):
            m = TabPFNClassifier(device="cpu", random_state=seed, ignore_pretraining_limits=True)
            m.fit(X.iloc[tr_i].to_numpy(), y.iloc[tr_i].to_numpy())
            oof[si, va_i] = m.predict_proba(X.iloc[va_i].to_numpy())[:, 1]
    return oof.mean(axis=0)


def run_reg(X, y, seeds=SEEDS, n_folds=N_FOLDS):
    oof = np.zeros((len(seeds), len(X)))
    for si, seed in enumerate(seeds):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr_i, va_i in kf.split(X):
            m = TabPFNRegressor(device="cpu", random_state=seed, ignore_pretraining_limits=True)
            m.fit(X.iloc[tr_i].to_numpy(), y.iloc[tr_i].to_numpy())
            oof[si, va_i] = m.predict(X.iloc[va_i].to_numpy())
    return oof.mean(axis=0)


def main():
    X, Xt, tr, te = load_xy()
    X, Xt = num(X), num(Xt)          # TabPFN wants numeric input
    y = tr["injured_in_risk_window"]

    results = {}

    print("=== TabPFN classifier: injured_in_risk_window (n=" + str(len(X))
          + ", d=" + str(X.shape[1]) + ") ===", flush=True)
    t0 = time.time()
    oof_c = run_clf(X, y)
    auc = roc_auc_score(y, oof_c)
    results["tabpfn_auc"] = float(auc)
    print("  AUC " + format(auc, ".5f") + "  (" + format(time.time() - t0, ".0f") + "s)",
          flush=True)

    inj = (y == 1).to_numpy()
    Xi = X[inj].reset_index(drop=True)
    for target, lo, hi in [("onset_day_offset", 1, 30), ("recovery_duration", 5, 20)]:
        yi = tr.loc[inj, target].reset_index(drop=True)
        print("\n=== TabPFN regressor: " + target + " (n=" + str(len(Xi)) + ") ===", flush=True)
        t0 = time.time()
        oof_r = run_reg(Xi, yi)
        mae = mean_absolute_error(yi, np.clip(oof_r, lo, hi))
        results["tabpfn_mae_" + target] = float(mae)
        print("  MAE " + format(mae, ".4f") + "  (" + format(time.time() - t0, ".0f") + "s)",
              flush=True)

    M = json.loads((REPORTS / "metrics.json").read_text())
    print("\n" + "=" * 60)
    print("  HEAD                     OURS       TabPFN     winner")
    print("=" * 60)
    a, b = M["auc_blend"], results["tabpfn_auc"]
    print("  injured (AUC, higher=better) " + format(a, ".4f") + "   " + format(b, ".4f")
          + "   " + ("ours" if a >= b else "TabPFN"))
    for t in ["onset_day_offset", "recovery_duration"]:
        a, b = M["mae_" + t], results["tabpfn_mae_" + t]
        print("  " + t.ljust(20) + " (MAE, lower=better) " + format(a, ".4f") + "   "
              + format(b, ".4f") + "   " + ("ours" if a <= b else "TabPFN"))
    print("=" * 60)

    (REPORTS / "bench_tabpfn.json").write_text(json.dumps(results, indent=2))
    print("wrote reports/bench_tabpfn.json")


if __name__ == "__main__":
    main()
