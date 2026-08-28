"""
Inference entry point: raw test CSVs -> submission.csv, using the persisted models.

    python predict.py                                  # uses ./data/test
    python predict.py --data-root /path/to/data --split test --out submission.csv

This is the "necessary files/code required to run or evaluate the models" half of
the Round-1 ZIP. It does not retrain anything: it rebuilds features with the same
leak-safe 30-day observation window used in training (src/features.py), loads the
artifacts written by src/build_final_models.py, applies the stored blend weights,
and thresholds with the stored rank-quantile rule.

Expected layout under --data-root:
    <split>/athlete_metadata.csv       <split>/sleepDay_merged.csv
    <split>/dailyActivity_merged.csv   <split>/training_sessions.csv
    <split>/hourlyHeartrate_merged.csv <split>/weightLogInfo_merged.csv
    <split>/hourlyIntensities_merged.csv
    <split>/hourlySteps_merged.csv     <split>/hourlyCalories_merged.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import features as F           # noqa: E402  (needs the sys.path line above)

MODELS = ROOT / "models"


def encode(X, sel, kind, cat_levels, cat_cols):
    """Reproduce the exact per-family encoding used at training time.

    XGBoost consumed `.cat.codes`, so the category level ORDER must match the
    order the training run used -- it is stored in the manifest for that reason.
    CatBoost consumed the raw strings.
    """
    A = X[sel].copy()
    cats = [c for c in cat_cols if c in sel]
    for c in cats:
        A[c] = pd.Categorical(A[c].astype(str), categories=cat_levels[c])
    if kind == "xgb":
        for c in cats:
            A[c] = A[c].cat.codes
    elif kind == "cat":
        for c in cats:
            A[c] = A[c].astype(str)
    return A


def blend_head(name, X, man, is_clf):
    """Weighted blend of the persisted models for one head."""
    head = man["heads"][name]
    sel, weights = head["features"], head["weights"]
    out = np.zeros(len(X))
    for kind, w in weights.items():
        if kind == "grp" or w <= 0:
            continue
        art = joblib.load(MODELS / (name + "_" + kind + ".joblib"))
        A = encode(X, sel, kind, man["cat_levels"], man["cat_cols"])
        p = art["model"].predict_proba(A)[:, 1] if is_clf else art["model"].predict(A)
        out += w * p
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(ROOT / "data"),
                    help="directory containing the <split>/ folder of raw CSVs")
    ap.add_argument("--split", default="test", help="subfolder name (default: test)")
    ap.add_argument("--out", default=str(ROOT / "submission.csv"))
    args = ap.parse_args()

    # ---- pre-flight: fail with instructions, not a stack trace ----------
    need = ["athlete_metadata.csv", "dailyActivity_merged.csv",
            "hourlyHeartrate_merged.csv", "hourlyIntensities_merged.csv",
            "hourlySteps_merged.csv", "sleepDay_merged.csv",
            "training_sessions.csv", "weightLogInfo_merged.csv"]
    split_dir = Path(args.data_root) / args.split
    if not split_dir.is_dir():
        sys.exit(
            "ERROR: no such directory: %s\n\n"
            "This ZIP does not bundle the competition CSVs. Point --data-root at\n"
            "the folder that CONTAINS the '%s' folder, e.g.\n"
            "    python predict.py --data-root /path/to/data --split %s\n"
            % (split_dir, args.split, args.split))
    missing = [f for f in need if not (split_dir / f).exists()]
    if missing:
        sys.exit("ERROR: %s is missing %d required file(s):\n  %s"
                 % (split_dir, len(missing), "\n  ".join(missing)))
    if not (MODELS / "manifest.json").exists():
        sys.exit("ERROR: models/manifest.json not found - run this from the "
                 "extracted ZIP root so that models/ sits alongside predict.py.")

    man = json.loads((MODELS / "manifest.json").read_text())

    # point features.py at the caller's data root and build from raw CSVs
    F.DATA = Path(args.data_root)
    print("building features from " + str(F.DATA / args.split) + " ...", flush=True)
    feat = F.build_features(args.split)
    ids = feat["Id"].to_numpy()
    X = feat.drop(columns=["Id"])

    lo_on, hi_on = man["onset_range"]
    lo_rc, hi_rc = man["recovery_range"]

    p_clf = blend_head("clf", X, man, is_clf=True)
    onset = np.clip(blend_head("onset", X, man, is_clf=False), lo_on, hi_on)
    recovery = blend_head("recovery", X, man, is_clf=False)

    # group median (sport+gender), with the training global median as fallback
    grp_w = man["heads"]["recovery"]["weights"].get("grp", 0.0)
    if grp_w > 0:
        g = joblib.load(MODELS / "recovery_grp.joblib")
        key = feat[g["keys"]].astype(str).agg("|".join, axis=1)
        recovery = recovery + grp_w * key.map(g["median_table"]).fillna(
            g["global_median"]).to_numpy()
    recovery = np.clip(recovery, lo_rc, hi_rc)

    # Threshold: a rank quantile, not an absolute probability. Task B penalises
    # a missed injury by 30 against baselines of ~7.6 / ~3.2, so recall is worth
    # far more than precision -- we flag everyone except the most confidently
    # healthy fraction. See src/rethreshold.py for the derivation.
    excl = man["threshold"]["exclusion_fraction"]
    thr = float(np.quantile(p_clf, excl)) if excl > 0 else -np.inf
    injured = (p_clf >= thr).astype(int)

    sub = pd.DataFrame({
        "athlete_id": ids,
        "injured_in_risk_window": injured,
        # required for EVERY athlete, including predicted-negatives (PDF page 3)
        "onset_day_offset": np.round(onset).astype(int),
        "recovery_duration": np.round(recovery).astype(int),
    })
    sub.to_csv(args.out, index=False)
    print("wrote " + args.out + "  rows=" + str(len(sub))
          + "  positive_rate=" + format(injured.mean(), ".4f"))


if __name__ == "__main__":
    main()
