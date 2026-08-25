"""
Patch for recovery_duration: oracle.py showed the strongest signal available
is the cross-fitted sport+gender group median (MAE 2.894), which beats the
3-model tree ensemble from final.py (MAE 2.926) -- confirmed by
recovery_max_residual_corr = 0.067 (~noise) once you condition on sport.

Adds that group median as a 4th blend candidate alongside lgb/xgb/cat, reusing
final.py's already-fitted machinery so this only touches recovery_duration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final import (load_xy, run_head, blend, conditional_mad_ceiling,
                   K_RECOVERY, SEEDS, N_FOLDS, ROOT, REPORTS)

TARGET = "recovery_duration"
LO, HI = 5, 20


def group_median_oof_test(df_inj, y, test_df, keys, seeds=SEEDS, n_folds=N_FOLDS):
    key_tr = df_inj[keys].astype(str).agg("|".join, axis=1)
    key_te = test_df[keys].astype(str).agg("|".join, axis=1)
    oof = np.zeros((len(seeds), len(y)))
    test_p = np.zeros((len(seeds), len(test_df)))
    for si, seed in enumerate(seeds):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr_i, va_i in kf.split(df_inj):
            med = y.iloc[tr_i].groupby(key_tr.iloc[tr_i].values).median()
            gl = y.iloc[tr_i].median()
            oof[si, va_i] = key_tr.iloc[va_i].map(med).fillna(gl).to_numpy()
            test_p[si] += (key_te.map(med).fillna(gl).to_numpy()) / n_folds
    return oof.mean(axis=0), test_p.mean(axis=0)


def main():
    X, Xt, tr, te = load_xy()
    y_cls = tr["injured_in_risk_window"]
    inj = (y_cls == 1).to_numpy()
    Xi = X[inj].reset_index(drop=True)
    yi = tr.loc[inj, TARGET].reset_index(drop=True)
    df_inj = tr.loc[inj].reset_index(drop=True)
    strat = pd.qcut(yi, 5, labels=False, duplicates="drop")

    print("=== recomputing tree OOF for " + TARGET + " (k=" + str(K_RECOVERY) + ") ===",
          flush=True)
    oof_r, test_r = run_head(Xi, yi, Xt, K_RECOVERY, False, strat)
    for kd, v in oof_r.items():
        mae = mean_absolute_error(yi, np.clip(v, LO, HI))
        print("  " + kd + " : MAE " + format(mae, ".4f"), flush=True)

    print("\n=== cross-fitted sport+gender group median ===", flush=True)
    grp_oof, grp_test = group_median_oof_test(df_inj, yi, te, ["sport", "gender"])
    mae_grp = mean_absolute_error(yi, np.clip(grp_oof, LO, HI))
    print("  grp : MAE " + format(mae_grp, ".4f"), flush=True)

    oof_r["grp"] = grp_oof
    test_r["grp"] = grp_test

    wr, mae_b = blend(oof_r, yi, mean_absolute_error, False)
    ob = np.column_stack([oof_r[k] for k in wr]) @ np.array(list(wr.values()))
    tb = np.column_stack([test_r[k] for k in wr]) @ np.array(list(wr.values()))
    mae_final = mean_absolute_error(yi, np.clip(ob, LO, HI))
    ceiling = conditional_mad_ceiling(ob, yi)
    print("\n  BLEND (+grp): MAE " + format(mae_final, ".4f") + "  weights " + str(wr),
          flush=True)

    M = json.loads((REPORTS / "metrics.json").read_text())
    old_mae = M["mae_" + TARGET]
    for kd in ("lgb", "xgb", "cat"):
        M["mae_" + TARGET + "_" + kd] = float(mean_absolute_error(yi, np.clip(oof_r[kd], LO, HI)))
    M["mae_" + TARGET + "_grp"] = float(mae_grp)
    M["mae_" + TARGET] = float(mae_final)
    M["blend_weights_" + TARGET] = {k: float(v) for k, v in wr.items()}
    M["oracle_mae_" + TARGET] = float(ceiling)
    (REPORTS / "metrics.json").write_text(json.dumps(M, indent=2))

    sub = pd.read_csv(ROOT / "submission.csv")
    sub[TARGET] = np.round(np.clip(tb, LO, HI)).astype(int)
    sub.to_csv(ROOT / "submission.csv", index=False)

    print("\n" + TARGET + ": " + format(old_mae, ".4f") + " -> " + format(mae_final, ".4f")
          + "  (ceiling " + format(ceiling, ".4f") + ")")
    print("wrote reports/metrics.json + submission.csv")


if __name__ == "__main__":
    main()
