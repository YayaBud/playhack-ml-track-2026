"""
Refit the final models on ALL training data and persist them to models/.

Everything else in this repo fits inside CV loops and throws the models away --
which is fine for honest scoring, but leaves nothing to ship. The Round-1 brief
asks for "a ZIP file containing their trained model(s) and the necessary
files/code required to run or evaluate the models", so this script produces the
actual artifacts, plus a manifest recording the feature lists, blend weights,
decision threshold and library versions needed to load them safely.

Model configs and per-head feature counts are imported from final.py rather
than re-declared, so the shipped artifacts cannot silently drift from the
cross-validated numbers in reports/metrics.json.

Also writes the test-set prediction arrays that rethreshold.py consumes:
  reports/final_test_clf.npy         classifier probabilities (blended)
  reports/final_test_onset.npy       onset predictions
  reports/final_test_recovery.npy    recovery predictions (trees + group median)
  reports/final_oof_recovery.npy     recovery OOF including the group median
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import lightgbm as lgb
import xgboost as xgb
import catboost

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from final import (load_xy, num, rank_cols, make_clf, make_reg,
                   K_CLF, K_ONSET, K_RECOVERY, SEEDS, ROOT, REPORTS)
from recovery_v2 import group_median_oof_test

MODELS = ROOT / "models"
SEED = SEEDS[0]
GROUP_KEYS = ["sport", "gender"]
ONSET_RANGE = (1, 30)
RECOVERY_RANGE = (5, 20)


def fit_one(kind, X, y, sel, is_clf, seed=SEED):
    """Fit a single model on the selected columns, matching final.py's
    per-family input encoding (CatBoost gets strings, XGBoost gets codes)."""
    m = make_clf(kind, seed) if is_clf else make_reg(kind, seed)
    if kind == "cat":
        cats = [c for c in F.CAT_COLS if c in sel]
        A = X[sel].copy()
        for c in cats:
            A[c] = A[c].astype(str)
        m.set_params(cat_features=cats)
        m.fit(A, y)
    elif kind == "xgb":
        m.fit(num(X[sel]), y, verbose=False)
    else:
        m.fit(X[sel], y)
    return m


def predict_one(kind, m, X, sel, is_clf):
    if kind == "cat":
        cats = [c for c in F.CAT_COLS if c in sel]
        A = X[sel].copy()
        for c in cats:
            A[c] = A[c].astype(str)
    elif kind == "xgb":
        A = num(X[sel])
    else:
        A = X[sel]
    return m.predict_proba(A)[:, 1] if is_clf else m.predict(A)


def build_head(name, X, y, Xt, k, is_clf, weights, artifacts):
    """Select features on the full training set, fit each family, blend."""
    sel = rank_cols(X, y, is_clf, SEED)[:k]
    print("  [" + name + "] selected " + str(len(sel)) + " features", flush=True)
    test = np.zeros(len(Xt))
    used = {}
    for kind, w in weights.items():
        if kind == "grp" or w <= 1e-6:        # grp handled separately; skip dead weights
            continue
        m = fit_one(kind, X, y, sel, is_clf)
        p = predict_one(kind, m, Xt, sel, is_clf)
        test += w * p
        path = MODELS / (name + "_" + kind + ".joblib")
        joblib.dump({"model": m, "features": sel, "kind": kind}, path, compress=3)
        used[kind] = float(w)
        print("    " + kind + " w=" + format(w, ".3f") + " -> " + path.name, flush=True)
    artifacts[name] = {"features": sel, "weights": used}
    return test


def main():
    MODELS.mkdir(exist_ok=True)
    # clear stale artifacts from the superseded src/pipeline.py
    for old in MODELS.glob("*.txt"):
        old.unlink()
        print("removed stale " + old.name)

    M = json.loads((REPORTS / "metrics.json").read_text())
    X, Xt, tr, te = load_xy()
    y_cls = tr["injured_in_risk_window"]
    inj = (y_cls == 1).to_numpy()
    Xi = X[inj].reset_index(drop=True)

    artifacts = {}
    print("=== classifier ===", flush=True)
    test_clf = build_head("clf", X, y_cls, Xt, K_CLF, True,
                          M["blend_weights_clf"], artifacts)

    print("=== onset_day_offset ===", flush=True)
    y_on = tr.loc[inj, "onset_day_offset"].reset_index(drop=True)
    test_on = build_head("onset", Xi, y_on, Xt, K_ONSET, False,
                         M["blend_weights_onset_day_offset"], artifacts)

    print("=== recovery_duration ===", flush=True)
    y_rc = tr.loc[inj, "recovery_duration"].reset_index(drop=True)
    w_rc = M["blend_weights_recovery_duration"]
    test_rc = build_head("recovery", Xi, y_rc, Xt, K_RECOVERY, False, w_rc, artifacts)

    # group median: fit on all injured athletes, and recompute its OOF so the
    # threshold sweep scores the same recovery blend we actually ship
    df_inj = tr.loc[inj].reset_index(drop=True)
    grp_oof, _ = group_median_oof_test(df_inj, y_rc, te, GROUP_KEYS)
    key_tr = df_inj[GROUP_KEYS].astype(str).agg("|".join, axis=1)
    med = y_rc.groupby(key_tr.values).median()
    global_med = float(y_rc.median())
    key_te = te[GROUP_KEYS].astype(str).agg("|".join, axis=1)
    grp_test = key_te.map(med).fillna(global_med).to_numpy()
    test_rc = test_rc + w_rc.get("grp", 0.0) * grp_test
    joblib.dump({"median_table": med.to_dict(), "global_median": global_med,
                 "keys": GROUP_KEYS}, MODELS / "recovery_grp.joblib", compress=3)
    artifacts["recovery"]["weights"]["grp"] = float(w_rc.get("grp", 0.0))
    print("    grp w=" + format(w_rc.get("grp", 0.0), ".3f") + " -> recovery_grp.joblib",
          flush=True)

    # recovery OOF including the group median (final.py saved trees only)
    tree_oof = np.load(REPORTS / "oof_recovery_duration.npy")
    tree_w = sum(v for k, v in w_rc.items() if k != "grp")
    oof_rc = tree_w * tree_oof + w_rc.get("grp", 0.0) * grp_oof
    np.save(REPORTS / "final_oof_recovery.npy", np.clip(oof_rc, *RECOVERY_RANGE))

    np.save(REPORTS / "final_test_clf.npy", test_clf)
    np.save(REPORTS / "final_test_onset.npy", np.clip(test_on, *ONSET_RANGE))
    np.save(REPORTS / "final_test_recovery.npy", np.clip(test_rc, *RECOVERY_RANGE))

    # XGBoost was trained on `.cat.codes`, which depend on the category ordering
    # that load_xy() built by unioning train and test levels. predict.py only
    # ever sees test data, so the exact level order has to travel with the
    # artifacts or the codes -- and therefore the predictions -- would differ.
    cat_levels = {c: [str(v) for v in X[c].cat.categories] for c in F.CAT_COLS}

    manifest = {
        "heads": artifacts,
        "cat_cols": F.CAT_COLS,
        "cat_levels": cat_levels,
        "onset_range": list(ONSET_RANGE),
        "recovery_range": list(RECOVERY_RANGE),
        "group_keys": GROUP_KEYS,
        "seed": SEED,
        "observation_window": [str(F.OBS_START.date()), str(F.OBS_END.date())],
        "threshold": None,          # filled in by rethreshold.py
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__, "catboost": catboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    (MODELS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(p.stat().st_size for p in MODELS.glob("*")) / 1e6
    print("\nwrote " + str(len(list(MODELS.glob('*.joblib')))) + " artifacts + manifest.json ("
          + format(total, ".1f") + " MB)")


if __name__ == "__main__":
    main()
