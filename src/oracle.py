"""
Bayes-ceiling ("oracle") estimate for the three PlayHack targets.

WHY AN ORACLE IS COMPUTABLE HERE
--------------------------------
Probing the data reverse-engineers the generator: injury is a *hazard* process
driven by a load-ramp latent that every ACWR-style feature measures.

  decile of steps_acwr    injury rate    mean onset
  bottom 8 deciles        0.16 - 0.29    ~22
  9th decile              0.70           12.4
  top decile              1.00            5.0

Two mechanisms: a low background hazard (injury lands late and near-randomly)
and an overload hazard that switches on above ACWR ~ 1.13 and drives injury
early and almost deterministically. The ACWR proxies correlate r ~ 0.99 *with
each other* but only ~0.86 with onset, so the latent is measured almost
perfectly and the residual is irreducible noise, not measurement error.

THE ORACLE IS ANALYTIC, NOT A FITTED SCORE
------------------------------------------
A fitted model can never hit AUC 1.0 here, because the label is a coin flip
with probability p(x): two athletes with identical covariates can land on
opposite labels. So the ceiling is not "AUC of the best fit" but the *expected*
AUC of ranking by the TRUE probability p:

    AUC* = [ SUM_ij p_i (1-p_j) 1(p_i > p_j) + 0.5 SUM_ij p_i (1-p_j) 1(p_i = p_j) ]
           / [ SUM_ij p_i (1-p_j) ]

We estimate p with a CROSS-FITTED isotonic regression on the latent (honest:
the conditional law is fitted on K-1 folds and applied to the held-out fold),
then evaluate the formula above. Same idea for the regression heads: under MAE
the Bayes act is the conditional MEDIAN, and the ceiling is the mean absolute
deviation of the conditional distribution around it.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
SEED = 42
N_FOLDS = 5


def load():
    tr = F.get_features("train")
    lab = pd.read_csv(F.DATA / "train" / "train_labels.csv")
    return tr.merge(lab, left_on="Id", right_on="athlete_id")


def build_latent(df, y):
    """1-D load-ramp latent.

    The `_acwr` columns are r~0.99 with one another, so PCA1 over just that
    tight family recovers the generator's driver with per-proxy jitter averaged
    out. Widening the family to every ramp-ish column dilutes it, so we keep it
    tight and verify against the best single proxy.
    """
    fam = [c for c in df.columns if c.endswith("_acwr")]
    M = df[fam].replace([np.inf, -np.inf], np.nan)
    M = SimpleImputer(strategy="median").fit_transform(M)
    M = StandardScaler().fit_transform(M)
    lat = PCA(n_components=1, random_state=SEED).fit_transform(M)[:, 0]
    if np.corrcoef(lat, y)[0, 1] < 0:
        lat = -lat

    # keep whichever ranks the label better: PCA1 or the single best proxy
    best_single, best_auc = None, 0.0
    for c in fam:
        x = df[c].replace([np.inf, -np.inf], np.nan)
        m = x.notna()
        if m.sum() < len(x) * 0.5:
            continue
        a = roc_auc_score(y[m], x[m])
        a = max(a, 1 - a)
        if a > best_auc:
            best_auc, best_single = a, c
    auc_pca = roc_auc_score(y, lat)
    print("  latent PCA1 over " + str(len(fam)) + " _acwr cols : AUC "
          + format(max(auc_pca, 1 - auc_pca), ".5f"))
    print("  best single proxy (" + str(best_single) + ")   : AUC " + format(best_auc, ".5f"))
    if best_auc > max(auc_pca, 1 - auc_pca):
        x = df[best_single].replace([np.inf, -np.inf], np.nan)
        lat = x.fillna(x.median()).to_numpy()
        if np.corrcoef(lat, y)[0, 1] < 0:
            lat = -lat
        print("  -> using best single proxy as latent")
    else:
        print("  -> using PCA1 as latent")
    return lat


def crossfit_isotonic(latent, y, n_folds=N_FOLDS):
    """Cross-fitted P(injured | latent). Isotonic = the monotone Bayes rule."""
    p = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for tr, va in skf.split(latent.reshape(-1, 1), y):
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(latent[tr], y.iloc[tr].to_numpy())
        p[va] = iso.predict(latent[va])
    return np.clip(p, 1e-6, 1 - 1e-6)


def expected_auc(p):
    """Expected AUC of ranking by the true probabilities p (closed form).

    Pairs are weighted by P(y_i=1, y_j=0) = p_i (1-p_j); ties score 0.5.
    O(n log n) via a sort plus prefix sums rather than the O(n^2) double loop.
    """
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    q = 1.0 - ps
    # for each i, weight of j with p_j < p_i  (strictly less) and p_j == p_i
    cum_q = np.concatenate([[0.0], np.cumsum(q)])
    total = ps.sum() * q.sum() - (ps * q).sum()   # exclude i == j

    concordant = 0.0
    ties = 0.0
    n = len(ps)
    i = 0
    while i < n:
        j = i
        while j < n and ps[j] == ps[i]:
            j += 1
        block = slice(i, j)
        # strictly-lower-scored negatives sit in [0, i)
        lower_q = cum_q[i]
        concordant += ps[block].sum() * lower_q
        # ties within the equal-score block (exclude self-pairing)
        blk_p, blk_q = ps[block].sum(), q[block].sum()
        ties += blk_p * blk_q - (ps[block] * q[block]).sum()
        i = j
    return (concordant + 0.5 * ties) / total


def knn_conditional(Z, y, k=60, n_folds=N_FOLDS):
    """Cross-fitted conditional median AND the conditional MAD around it."""
    med = np.zeros(len(y))
    mad = np.zeros(len(y))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    yv = y.to_numpy()
    for tr, va in kf.split(Z):
        nn = NearestNeighbors(n_neighbors=min(k, len(tr))).fit(Z[tr])
        idx = nn.kneighbors(Z[va], return_distance=False)
        neigh = yv[tr][idx]
        m = np.median(neigh, axis=1)
        med[va] = m
        mad[va] = np.abs(neigh - m[:, None]).mean(axis=1)
    return med, mad


def group_conditional_median(df, y, keys, n_folds=N_FOLDS):
    """Cross-fitted median of y within categorical cells."""
    oof = np.full(len(y), np.nan)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    key_all = df[keys].astype(str).agg("|".join, axis=1)
    for tr, va in kf.split(df):
        med = y.iloc[tr].groupby(key_all.iloc[tr].values).median()
        gl = y.iloc[tr].median()
        oof[va] = key_all.iloc[va].map(med).fillna(gl).to_numpy()
    return oof


def main():
    df = load()
    y = df["injured_in_risk_window"]
    print("=== recovering the load-ramp latent ===")
    latent = build_latent(df, y)

    out = {}

    # ---- 1. classification ceiling -----------------------------------------
    p = crossfit_isotonic(latent, y)
    auc_fit = roc_auc_score(y, p)
    auc_star = expected_auc(p)
    out["achievable_auc_isotonic_on_latent"] = float(auc_fit)
    out["oracle_auc"] = float(auc_star)
    print("\nORACLE  injured_in_risk_window")
    print("  cross-fitted isotonic on latent     : AUC " + format(auc_fit, ".5f"))
    print("  ORACLE  E[AUC | true p] (analytic)  : AUC " + format(auc_star, ".5f"))
    print("    p ranges " + format(p.min(), ".3f") + " .. " + format(p.max(), ".3f")
          + ", mean " + format(p.mean(), ".3f"))
    # best attainable accuracy / F1 given p
    thr_grid = np.linspace(0.05, 0.95, 181)
    best_f1, best_thr = 0.0, 0.5
    for t in thr_grid:
        pred = (p >= t)
        tp = (p * pred).sum()
        fp = ((1 - p) * pred).sum()
        fn = (p * ~pred).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    out["oracle_f1"] = float(best_f1)
    out["oracle_f1_threshold"] = float(best_thr)
    print("  ORACLE  E[F1] at best threshold     : F1  " + format(best_f1, ".5f")
          + " @ " + format(best_thr, ".2f"))

    # ---- 2. onset ceiling ---------------------------------------------------
    inj = (y == 1).to_numpy()
    y_on = df.loc[inj, "onset_day_offset"].reset_index(drop=True)
    Zon = latent[inj].reshape(-1, 1)
    med_on, mad_on = knn_conditional(Zon, y_on, k=60)
    out["constant_mae_onset"] = float(mean_absolute_error(y_on, np.full(len(y_on), y_on.median())))
    out["achievable_mae_onset_knn"] = float(mean_absolute_error(y_on, np.clip(med_on, 1, 30)))
    out["oracle_mae_onset"] = float(mad_on.mean())
    print("\nORACLE  onset_day_offset  (injured only, n=" + str(len(y_on)) + ")")
    print("  constant-median baseline            : MAE " + format(out["constant_mae_onset"], ".4f"))
    print("  cross-fitted kNN median on latent   : MAE " + format(out["achievable_mae_onset_knn"], ".4f"))
    print("  ORACLE  E[|y - median|] (cond. MAD) : MAE " + format(out["oracle_mae_onset"], ".4f"))

    # ---- 3. recovery ceiling ------------------------------------------------
    y_rc = df.loc[inj, "recovery_duration"].reset_index(drop=True)
    dfi = df[inj].reset_index(drop=True)
    out["constant_mae_recovery"] = float(
        mean_absolute_error(y_rc, np.full(len(y_rc), y_rc.median())))
    print("\nORACLE  recovery_duration (injured only)")
    print("  constant-median baseline            : MAE "
          + format(out["constant_mae_recovery"], ".4f"))
    best_rc, best_keys = out["constant_mae_recovery"], ("constant",)
    for keys in [["sport"], ["position"], ["sport", "gender"], ["position", "gender"]]:
        m = group_conditional_median(dfi, y_rc, keys)
        mae = mean_absolute_error(y_rc, m)
        out["mae_recovery_by_" + "+".join(keys)] = float(mae)
        print("  cross-fitted median by " + "+".join(keys).ljust(16)
              + ": MAE " + format(mae, ".4f"))
        if mae < best_rc:
            best_rc, best_keys = mae, tuple(keys)
    # conditional MAD within the winning cells = the ceiling
    key_all = dfi[list(best_keys)].astype(str).agg("|".join, axis=1) \
        if best_keys != ("constant",) else pd.Series(["all"] * len(y_rc))
    mad_rc = y_rc.groupby(key_all.values).transform(lambda s: (s - s.median()).abs().mean())
    out["oracle_mae_recovery"] = float(mad_rc.mean())
    print("  ORACLE  E[|y - median|] within " + "+".join(best_keys)
          + " : MAE " + format(out["oracle_mae_recovery"], ".4f"))

    resid = y_rc - group_conditional_median(dfi, y_rc, ["position"])
    num = dfi.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    num = num.drop(columns=["Id", "athlete_id", "recovery_duration",
                            "onset_day_offset", "injured_in_risk_window"], errors="ignore")
    top = num.corrwith(pd.Series(resid)).dropna().abs().sort_values(ascending=False)
    out["recovery_max_residual_corr"] = float(top.iloc[0])
    print("  max |corr| of any feature vs residual: " + format(top.iloc[0], ".4f")
          + " (" + top.index[0] + ")  -> noise, nothing left to model")

    (REPORTS / "oracle.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/oracle.json")
    return out


if __name__ == "__main__":
    main()
