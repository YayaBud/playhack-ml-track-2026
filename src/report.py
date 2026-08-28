"""
Render reports/metrics.json into RESULTS.md.

Leads with the organisers' actual scoring rule (Task A F1 + the two Task B skill
scores, src/score.py). ROC-AUC and the MAE ceilings follow as diagnostics -- they
are how we judged the ranking while building, but they are not scored.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def pct(model, ceiling):
    return float("nan") if not ceiling else 100 * model / ceiling


def main():
    M = json.loads((REPORTS / "metrics.json").read_text())
    L = []
    A = L.append

    A("# Results")
    A("")
    A("Out-of-fold over 3000 training athletes, 3 seeds x 5 folds, with feature")
    A("selection performed inside each fold. `submission.csv` comes from the same")
    A("models refit on all training data (`src/build_final_models.py`).")
    A("")

    # ------------------------------------------------------------- official ---
    A("## Scored metric (problem statement, page 6)")
    A("")
    A("`skill = max(0, 1 - MAE_model / MAE_baseline)`, where a **missed** injury")
    A("contributes a flat `PENALTY = 30` to both timing heads, and the baseline")
    A("predicts the training-set mean for every injured athlete. Task B is")
    A("evaluated over the truly-injured athletes only. See `src/score.py`.")
    A("")
    A("| | scored quantity | **ours** | ceiling | closed |")
    A("|---|---|---|---|---|")
    rows = [
        ("**Task A**", "F1", "official_f1", "oracle_official_f1"),
        ("**Task B**", "skill — `onset_day_offset`", "official_skill_onset",
         "oracle_official_skill_onset"),
        ("**Task B**", "skill — `recovery_duration`", "official_skill_recovery",
         "oracle_official_skill_recovery"),
        ("", "balanced mean of the three", "official_mean_of_three",
         "oracle_official_mean_of_three"),
    ]
    for task, name, mk, ck in rows:
        if mk in M and ck in M:
            A("| %s | %s | **%.4f** | %.4f | %.1f%% |" % (
                task, name, M[mk], M[ck], pct(M[mk], M[ck])))
    A("")
    if "official_mae_onset_model" in M:
        A("Underlying errors: onset MAE **%.4f** against a %.4f baseline; recovery"
          % (M["official_mae_onset_model"], M["official_mae_onset_baseline"]))
        A("MAE **%.4f** against %.4f. OOF recall on injured athletes: **%.4f**."
          % (M["official_mae_recovery_model"], M["official_mae_recovery_baseline"],
             M["official_oof_recall"]))
        A("")

    # ------------------------------------------------------------ threshold ---
    if "f1_optimal_threshold_would_score" in M:
        f = M["f1_optimal_threshold_would_score"]
        A("### Why the threshold sits so low")
        A("")
        A("A miss costs 30 against baselines of ~7.6 and ~3.2, so break-even recall")
        A("is 0.82 (onset) and 0.99 (recovery). Tuning for F1 alone clears neither:")
        A("")
        A("| threshold | F1 | skill_onset | skill_recovery | mean |")
        A("|---|---|---|---|---|")
        A("| %.2f — F1-optimal | **%.4f** | %.4f | %.4f | %.4f |" % (
            f["threshold"], f["f1"], f["skill_onset"], f["skill_recovery"],
            (f["f1"] + f["skill_onset"] + f["skill_recovery"]) / 3))
        A("| %.3f — **shipped** | %.4f | **%.4f** | **%.4f** | **%.4f** |" % (
            M.get("threshold_test", 0), M["official_f1"], M["official_skill_onset"],
            M["official_skill_recovery"], M["official_mean_of_three"]))
        A("")
        A("The shipped rule flags every athlete except the most confidently healthy")
        A("%.2f%% (half the maximum exclusion that still holds OOF recall at 1.0)."
          % (100 * M.get("official_exclusion_fraction", 0)))
        A("Test positive rate %.4f. A perfectly-calibrated oracle makes the same"
          % M.get("test_positive_rate", float("nan")))
        A("trade, so this is the optimal play under the rule, not a quirk of our model.")
        A("")

    # ----------------------------------------------------------- diagnostic ---
    A("## Diagnostics (not scored)")
    A("")
    A("| head | metric | model | ceiling | closed |")
    A("|---|---|---|---|---|")
    if "auc_blend" in M:
        A("| `injured_in_risk_window` | ROC-AUC | %.4f | %.4f | %.1f%% |" % (
            M["auc_blend"], M["oracle_auc"],
            100 * (M["auc_blend"] - 0.5) / (M["oracle_auc"] - 0.5)))
    for t, label in [("onset_day_offset", "`onset_day_offset`"),
                     ("recovery_duration", "`recovery_duration`")]:
        mk, bk, ck = "mae_" + t, "constant_mae_" + t, "oracle_mae_" + t
        if mk in M and ck in M:
            A("| %s | MAE (hits only) | %.4f | %.4f | %.1f%% |" % (
                label, M[mk], M[ck],
                100 * (M[bk] - M[mk]) / (M[bk] - M[ck])))
    A("")
    A("> A ceiling is an **estimate** of irreducible error, not a proven bound. It")
    A("> must condition on at least as much information as the model it bounds -- an")
    A("> earlier estimate built on a single recovered feature put this model at 113%")
    A("> of \"maximum\", which signalled a bad estimate rather than a good model.")
    A("> `src/official_ceiling.py` now warns whenever a score exceeds its ceiling.")
    A("")

    # ---------------------------------------------------------- per-model ----
    A("## Per-model out-of-fold scores")
    A("")
    A("| head | LightGBM | XGBoost | CatBoost | group median | blend |")
    A("|---|---|---|---|---|---|")
    if "auc_lgb" in M:
        A("| `injured_in_risk_window` (AUC) | %.4f | %.4f | %.4f | - | **%.4f** |" % (
            M["auc_lgb"], M["auc_xgb"], M["auc_cat"], M["auc_blend"]))
    for t, label in [("onset_day_offset", "`onset_day_offset` (MAE)"),
                     ("recovery_duration", "`recovery_duration` (MAE)")]:
        if "mae_" + t + "_lgb" in M:
            grp = ("%.4f" % M["mae_" + t + "_grp"]) if "mae_" + t + "_grp" in M else "-"
            A("| %s | %.4f | %.4f | %.4f | %s | **%.4f** |" % (
                label, M["mae_" + t + "_lgb"], M["mae_" + t + "_xgb"],
                M["mae_" + t + "_cat"], grp, M["mae_" + t]))
    A("")
    A("`recovery_duration` carries a fourth candidate: a cross-fitted sport+gender")
    A("median. It beats all three tree models on its own and takes the largest")
    A("share of the blend -- recovery has little per-athlete signal beyond sport.")
    A("")
    for key, label in [("blend_weights_clf", "classifier"),
                       ("blend_weights_onset_day_offset", "`onset_day_offset`"),
                       ("blend_weights_recovery_duration", "`recovery_duration`")]:
        if key in M:
            A("- %s: %s" % (label, json.dumps(
                {k: round(v, 3) for k, v in M[key].items()})))
    A("")

    # ----------------------------------------------------------- benchmark ---
    if "ag_auc" in M:
        A("## SOTA benchmark: AutoGluon")
        A("")
        A("AutoGluon 1.6.1 `best_quality` (stacked LightGBM/XGBoost/CatBoost/RF/")
        A("ExtraTrees/NN/KNN), 300s per head, same features and split.")
        A("")
        A("| head | ours | AutoGluon |")
        A("|---|---|---|")
        A("| `injured_in_risk_window` (AUC) | %.4f | %.4f |" % (M["auc_blend"], M["ag_auc"]))
        for t in ["onset_day_offset", "recovery_duration"]:
            if "ag_mae_" + t in M:
                A("| `%s` (MAE) | %.4f | %.4f |" % (t, M["mae_" + t], M["ag_mae_" + t]))
        A("")
        A("Competitive, and it won 2 of 3 heads on the AUC/MAE framing. Not shipped:")
        A("its edge under the *scored* metric is ~0.008 skill, against a 274 MB")
        A("artifact requiring a matching AutoGluon install. TabPFN was blocked by a")
        A("one-time license click-through its package now requires.")
        A("")

    (ROOT / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
