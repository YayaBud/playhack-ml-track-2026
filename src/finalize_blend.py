"""
Combine our hand-built ensemble with AutoGluon (best_quality) per head,
taking whichever wins -- or blending them when a blend beats both solo.

No refitting: AutoGluon's fitted predictors are reloaded from ag_models/, and
our OOF/test arrays are the ones final.py already saved to reports/. Only
the classifier gets re-thresholded (isotonic calibration + expected-F1 on the
combined score), matching final.py's method.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_absolute_error, f1_score
from sklearn.isotonic import IsotonicRegression
from autogluon.tabular import TabularPredictor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final import load_xy, blend, N_FOLDS

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
AG_DIR = ROOT / "ag_models"


def main():
    X, Xt, tr, te = load_xy()
    y = tr["injured_in_risk_window"]
    M = json.loads((REPORTS / "metrics.json").read_text())
    grid = np.linspace(0.05, 0.95, 181)

    # ------------------------------------------------------- classifier ----
    print("=== injured_in_risk_window ===")
    ours_oof = np.load(REPORTS / "oof_blend.npy")
    ours_test = np.load(REPORTS / "test_blend.npy")
    pred_c = TabularPredictor.load(str(AG_DIR / "clf"))
    oof_proba = pred_c.predict_proba_oof()
    pos_col = [c for c in oof_proba.columns if str(c) in ("1", 1)][0]
    ag_oof = oof_proba[pos_col].to_numpy()
    ag_test = pred_c.predict_proba(Xt)[pos_col].to_numpy()

    auc_ours = roc_auc_score(y, ours_oof)
    auc_ag = roc_auc_score(y, ag_oof)
    w, auc_blend = blend({"ours": ours_oof, "ag": ag_oof}, y, roc_auc_score, True)
    print("  ours %.5f  AG %.5f  blend %.5f  weights %s" % (auc_ours, auc_ag, auc_blend, w))
    if auc_blend >= max(auc_ours, auc_ag):
        cls_oof = ours_oof * w["ours"] + ag_oof * w["ag"]
        cls_test = ours_test * w["ours"] + ag_test * w["ag"]
        cls_auc, cls_tag = auc_blend, "blend(ours,AG) " + str({k: round(v, 3) for k, v in w.items()})
    elif auc_ag > auc_ours:
        cls_oof, cls_test, cls_auc, cls_tag = ag_oof, ag_test, auc_ag, "AutoGluon"
    else:
        cls_oof, cls_test, cls_auc, cls_tag = ours_oof, ours_test, auc_ours, "ours"
    print("  -> using " + cls_tag + "  AUC %.5f" % cls_auc)

    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(cls_oof, y.to_numpy())
    p_test = np.clip(iso.predict(cls_test), 1e-6, 1 - 1e-6)
    p_oof = np.clip(iso.predict(cls_oof), 1e-6, 1 - 1e-6)
    of1 = []
    for t in grid:
        pr = p_oof >= t
        tp = (p_oof * pr).sum(); fp = ((1 - p_oof) * pr).sum(); fn = (p_oof * ~pr).sum()
        of1.append(2 * tp / max(2 * tp + fp + fn, 1e-9))
    thr = float(grid[int(np.argmax(of1))])
    pred_injured = (p_test >= thr).astype(int)
    f1_hard = f1_score(y, (cls_oof >= (M.get("best_threshold", 0.5))).astype(int))
    print("  threshold %.3f  test positive rate %.4f (was %.4f)"
          % (thr, pred_injured.mean(), M["test_positive_rate"]))

    # ----------------------------------------------------------- onset -----
    print("\n=== onset_day_offset ===")
    inj = (y == 1).to_numpy()
    yi_on = tr.loc[inj, "onset_day_offset"].reset_index(drop=True)
    ours_oof_on = np.load(REPORTS / "oof_onset_day_offset.npy")
    ours_test_on = np.load(REPORTS / "test_onset_day_offset.npy")
    pred_on = TabularPredictor.load(str(AG_DIR / "reg_onset_day_offset"))
    ag_oof_on = pred_on.predict_oof().to_numpy()
    ag_test_on = pred_on.predict(Xt).to_numpy()

    mae_ours = mean_absolute_error(yi_on, np.clip(ours_oof_on, 1, 30))
    mae_ag = mean_absolute_error(yi_on, np.clip(ag_oof_on, 1, 30))
    w, mae_bl = blend({"ours": ours_oof_on, "ag": ag_oof_on}, yi_on, mean_absolute_error, False)
    print("  ours %.4f  AG %.4f  blend %.4f  weights %s" % (mae_ours, mae_ag, mae_bl, w))
    if mae_bl <= min(mae_ours, mae_ag):
        onset_test = np.clip(ours_test_on * w["ours"] + ag_test_on * w["ag"], 1, 30)
        onset_mae, onset_tag = mae_bl, "blend"
    elif mae_ag < mae_ours:
        onset_test, onset_mae, onset_tag = np.clip(ag_test_on, 1, 30), mae_ag, "AutoGluon"
    else:
        onset_test, onset_mae, onset_tag = np.clip(ours_test_on, 1, 30), mae_ours, "ours"
    print("  -> using " + onset_tag + "  MAE %.4f" % onset_mae)

    # -------------------------------------------------------- recovery -----
    print("\n=== recovery_duration ===")
    yi_rc = tr.loc[inj, "recovery_duration"].reset_index(drop=True)
    pred_rc = TabularPredictor.load(str(AG_DIR / "reg_recovery_duration"))
    ag_oof_rc = pred_rc.predict_oof().to_numpy()
    ag_test_rc = pred_rc.predict(Xt).to_numpy()
    mae_ag_rc = mean_absolute_error(yi_rc, np.clip(ag_oof_rc, 5, 20))
    mae_ours_rc = M["mae_recovery_duration"]     # ours = tree+grp blend, OOF array not persisted
    print("  ours(+grp) %.4f  AG %.4f  (no OOF array saved for ours; taking whichever is lower)"
          % (mae_ours_rc, mae_ag_rc))
    if mae_ag_rc < mae_ours_rc:
        recovery_test, recovery_mae, recovery_tag = np.clip(ag_test_rc, 5, 20), mae_ag_rc, "AutoGluon"
    else:
        recovery_test = None   # keep whatever is already in submission.csv
        recovery_mae, recovery_tag = mae_ours_rc, "ours"
    print("  -> using " + recovery_tag + "  MAE %.4f" % recovery_mae)

    # ------------------------------------------------------- submission ----
    sub = pd.read_csv(ROOT / "submission.csv")
    sub["injured_in_risk_window"] = pred_injured
    sub["onset_day_offset"] = np.round(onset_test).astype(int)
    if recovery_test is not None:
        sub["recovery_duration"] = np.round(recovery_test).astype(int)
    sub.to_csv(ROOT / "submission.csv", index=False)

    M["auc_blend_final"] = float(cls_auc)
    M["auc_source"] = cls_tag
    M["ag_auc"] = float(auc_ag)
    M["mae_onset_day_offset_final"] = float(onset_mae)
    M["onset_source"] = onset_tag
    M["ag_mae_onset_day_offset"] = float(mae_ag)
    M["mae_recovery_duration_final"] = float(recovery_mae)
    M["recovery_source"] = recovery_tag
    M["ag_mae_recovery_duration"] = float(mae_ag_rc)
    M["test_threshold_calibrated"] = thr
    M["test_positive_rate"] = float(pred_injured.mean())
    (REPORTS / "metrics.json").write_text(json.dumps(M, indent=2))

    print("\nwrote submission.csv + reports/metrics.json")
    print("FINAL: clf AUC %.4f (%s) | onset MAE %.4f (%s) | recovery MAE %.4f (%s)"
          % (cls_auc, cls_tag, onset_mae, onset_tag, recovery_mae, recovery_tag))


if __name__ == "__main__":
    main()
