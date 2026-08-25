"""
PlayHack ML Track 2026 (IIT Guwahati) - injury risk pipeline.

Data has a 60-day train / 30-day test split: days 1-30 are the observation
window (available in both splits), days 31-60 (train only) are the risk
window being predicted. All features below are built ONLY from days 1-30
to avoid leaking the risk window into training.

Targets (from train_labels.csv / sample_submission.csv):
  injured_in_risk_window  - binary classification
  onset_day_offset        - regression, defined only when injured=1 (1..30)
  recovery_duration       - regression, defined only when injured=1 (5..20)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_absolute_error
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

OBS_END = "2026-02-03"  # last day of the 30-day observation window (inclusive)
CAT_COLS = ["sport", "gender", "dominant_side", "position"]
SEED = 42


def _slope(y):
    y = np.asarray(y, dtype=float)
    if len(y) < 2 or np.all(np.isnan(y)):
        return np.nan
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    return np.polyfit(x[mask], y[mask], 1)[0]


def load_metadata(split):
    df = pd.read_csv(DATA / split / "athlete_metadata.csv")
    return df.rename(columns={"athlete_id": "Id"})


def daily_features(split):
    df = pd.read_csv(DATA / split / "dailyActivity_merged.csv", parse_dates=["ActivityDate"])
    if split == "train":
        df = df[df["ActivityDate"] <= OBS_END]
    g = df.groupby("Id")
    out = pd.DataFrame({
        "steps_mean": g["TotalSteps"].mean(),
        "steps_std": g["TotalSteps"].std(),
        "steps_sum": g["TotalSteps"].sum(),
        "steps_slope": g["TotalSteps"].apply(_slope),
        "distance_mean": g["TotalDistance"].mean(),
        "very_active_min_mean": g["VeryActiveMinutes"].mean(),
        "fairly_active_min_mean": g["FairlyActiveMinutes"].mean(),
        "lightly_active_min_mean": g["LightlyActiveMinutes"].mean(),
        "sedentary_min_mean": g["SedentaryMinutes"].mean(),
        "calories_mean": g["Calories"].mean(),
        "calories_std": g["Calories"].std(),
        "calories_slope": g["Calories"].apply(_slope),
        "n_days_logged": g["ActivityDate"].count(),
    })
    return out


def sleep_features(split):
    df = pd.read_csv(DATA / split / "sleepDay_merged.csv", parse_dates=["SleepDay"])
    if split == "train":
        df = df[df["SleepDay"] <= OBS_END]
    df["efficiency"] = df["TotalMinutesAsleep"] / df["TotalTimeInBed"].replace(0, np.nan)
    df["debt"] = df["TotalTimeInBed"] - df["TotalMinutesAsleep"]
    g = df.groupby("Id")
    out = pd.DataFrame({
        "sleep_min_mean": g["TotalMinutesAsleep"].mean(),
        "sleep_min_std": g["TotalMinutesAsleep"].std(),
        "time_in_bed_mean": g["TotalTimeInBed"].mean(),
        "sleep_efficiency_mean": g["efficiency"].mean(),
        "sleep_debt_mean": g["debt"].mean(),
        "n_sleep_records": g["SleepDay"].count(),
    })
    return out


def weight_features(split):
    df = pd.read_csv(DATA / split / "weightLogInfo_merged.csv", parse_dates=["Date"])
    if split == "train":
        df = df[df["Date"] <= OBS_END]
    df = df.sort_values("Date")
    last = df.groupby("Id").last()[["WeightKg", "Fat", "BMI"]]
    last.columns = ["weight_last", "fat_last", "bmi_last"]
    n = df.groupby("Id").size().rename("n_weight_logs")
    out = last.join(n, how="outer")
    return out


def session_features(split):
    df = pd.read_csv(DATA / split / "training_sessions.csv", parse_dates=["date"])
    df = df.rename(columns={"athlete_id": "Id"})
    df["hours"] = df["end_hour"] - df["start_hour"]
    window_end = pd.Timestamp(OBS_END)
    window_start = window_end - pd.Timedelta(days=29)
    df = df[(df["date"] >= window_start) & (df["date"] <= window_end)]

    g = df.groupby("Id")
    type_counts = df.pivot_table(index="Id", columns="sport_session_type", values="session_id",
                                  aggfunc="count", fill_value=0)
    type_counts.columns = [f"n_{c}" for c in type_counts.columns]

    daily_hours = df.groupby(["Id", "date"])["hours"].sum().reset_index()
    acute_cut = window_end - pd.Timedelta(days=6)   # last 7 days
    acute = daily_hours[daily_hours["date"] >= acute_cut].groupby("Id")["hours"].sum() / 7.0
    chronic = daily_hours.groupby("Id")["hours"].sum() / 30.0
    acwr = (acute / chronic.replace(0, np.nan)).rename("acwr")

    out = pd.DataFrame({
        "n_sessions_total": g.size(),
        "total_hours": g["hours"].sum(),
        "n_unique_session_days": g["date"].nunique(),
        "max_sessions_per_day": df.groupby(["Id", "date"]).size().groupby("Id").max(),
    })
    out = out.join(type_counts, how="left").join(acwr, how="left")
    out["rest_days"] = 30 - out["n_unique_session_days"]
    return out


def _first_last_delta(daily, id_col="Id"):
    """Given a per-Id-per-day series (indexed by [Id, day]), return mean/std/slope
    plus a first-week vs last-week delta (fatigue/trend signal)."""
    g = daily.groupby(level=0)
    slope = g.apply(lambda s: _slope(s.values))
    mean_ = g.mean()
    std_ = g.std()

    def first_last(s):
        n = len(s)
        if n < 4:
            return np.nan
        k = max(1, n // 4)
        return s.iloc[-k:].mean() - s.iloc[:k].mean()

    delta = g.apply(first_last)
    return mean_, std_, slope, delta


def hourly_features(split):
    hr = pd.read_csv(DATA / split / "hourlyHeartrate_merged.csv",
                      usecols=["Id", "ActivityHour", "AvgHeartRate"],
                      dtype={"Id": "int32", "AvgHeartRate": "float32"})
    ints = pd.read_csv(DATA / split / "hourlyIntensities_merged.csv",
                        usecols=["Id", "ActivityHour", "TotalIntensity"],
                        dtype={"Id": "int32", "TotalIntensity": "float32"})
    steps = pd.read_csv(DATA / split / "hourlySteps_merged.csv",
                         usecols=["Id", "ActivityHour", "StepTotal"],
                         dtype={"Id": "int32", "StepTotal": "float32"})

    hr["day"] = pd.to_datetime(hr["ActivityHour"], format="%Y-%m-%dT%H:%M:%S").dt.date
    ints_dt = pd.to_datetime(ints["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
    steps_dt = pd.to_datetime(steps["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
    ints["day"] = ints_dt.dt.date
    steps["day"] = steps_dt.dt.date

    if split == "train":
        cutoff = pd.to_datetime(OBS_END).date()
        hr = hr[hr["day"] <= cutoff]
        ints = ints[ints_dt <= pd.to_datetime(OBS_END) + pd.Timedelta(hours=23, minutes=59)]
        steps = steps[steps_dt <= pd.to_datetime(OBS_END) + pd.Timedelta(hours=23, minutes=59)]

    hr_g = hr.groupby("Id")["AvgHeartRate"]
    ints_g = ints.groupby("Id")["TotalIntensity"]
    steps_g = steps.groupby("Id")["StepTotal"]

    # resting HR proxy: bottom-quartile hourly HR per athlete (lowest-activity hours)
    resting_hr = hr.groupby("Id")["AvgHeartRate"].quantile(0.1).rename("resting_hr_p10")

    # daily aggregates -> trend/fatigue deltas (resting HR drift is a classic overtraining marker)
    daily_hr = hr.groupby(["Id", "day"])["AvgHeartRate"].mean()
    hr_mean, hr_std_daily, hr_slope, hr_delta = _first_last_delta(daily_hr)

    daily_intensity = ints.groupby(["Id", "day"])["TotalIntensity"].sum()
    int_mean, int_std_daily, int_slope, int_delta = _first_last_delta(daily_intensity)

    daily_steps_hourly_std = steps.groupby(["Id", "day"])["StepTotal"].std().groupby(level=0).mean()

    out = pd.DataFrame({
        "hr_mean": hr_g.mean(),
        "hr_std": hr_g.std(),
        "hr_max": hr_g.max(),
        "resting_hr_p10": resting_hr,
        "hr_daily_slope": hr_slope,
        "hr_first_last_delta": hr_delta,
        "intensity_mean": ints_g.mean(),
        "intensity_std": ints_g.std(),
        "intensity_daily_slope": int_slope,
        "intensity_first_last_delta": int_delta,
        "steps_hourly_std": steps_g.std(),
        "steps_within_day_std_mean": daily_steps_hourly_std,
    })
    return out


def build_features(split):
    meta = load_metadata(split)
    parts = [daily_features(split), sleep_features(split), weight_features(split),
             session_features(split), hourly_features(split)]
    df = meta.set_index("Id")
    for p in parts:
        df = df.join(p, how="left")
    df = df.drop(columns=["team_id"], errors="ignore")
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    return df.reset_index()


def get_or_build_features(split):
    cache = DATA / f"features_{split}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = build_features(split)
    df.to_parquet(cache)
    return df


def train_and_predict():
    train_feat = get_or_build_features("train")
    test_feat = get_or_build_features("test")
    labels = pd.read_csv(DATA / "train" / "train_labels.csv")

    train_df = train_feat.merge(labels, left_on="Id", right_on="athlete_id")
    feature_cols = [c for c in train_feat.columns if c != "Id"]
    for c in CAT_COLS:
        test_feat[c] = test_feat[c].astype("category")

    X = train_df[feature_cols]
    y_cls = train_df["injured_in_risk_window"]
    X_test = test_feat[feature_cols]

    metrics = {}

    # ---- classifier: injured_in_risk_window ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(X))
    test_pred_cls = np.zeros(len(X_test))
    clf_params = dict(objective="binary", n_estimators=400, learning_rate=0.03,
                       num_leaves=15, min_child_samples=20, subsample=0.8,
                       colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_cls)):
        model = lgb.LGBMClassifier(**clf_params)
        model.fit(X.iloc[tr_idx], y_cls.iloc[tr_idx],
                  eval_set=[(X.iloc[va_idx], y_cls.iloc[va_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_pred[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        test_pred_cls += model.predict_proba(X_test)[:, 1] / skf.n_splits
    auc = roc_auc_score(y_cls, oof_pred)
    metrics["classifier_oof_roc_auc"] = auc
    final_clf = lgb.LGBMClassifier(**clf_params)
    final_clf.fit(X, y_cls)
    final_clf.booster_.save_model(str(MODELS / "classifier.txt"))

    # ---- regressors: onset_day_offset, recovery_duration (injured rows only) ----
    inj_mask = y_cls == 1
    X_inj = X[inj_mask]
    reg_params = dict(objective="regression", n_estimators=300, learning_rate=0.03,
                       num_leaves=7, min_child_samples=10, subsample=0.8,
                       colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1)
    reg_preds_test = {}
    for target in ["onset_day_offset", "recovery_duration"]:
        y_reg = train_df.loc[inj_mask, target]
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.zeros(len(X_inj))
        for tr_idx, va_idx in kf.split(X_inj):
            m = lgb.LGBMRegressor(**reg_params)
            m.fit(X_inj.iloc[tr_idx], y_reg.iloc[tr_idx])
            oof[va_idx] = m.predict(X_inj.iloc[va_idx])
        mae = mean_absolute_error(y_reg, oof)
        metrics[f"{target}_oof_mae"] = mae
        final_reg = lgb.LGBMRegressor(**reg_params)
        final_reg.fit(X_inj, y_reg)
        final_reg.booster_.save_model(str(MODELS / f"{target}_regressor.txt"))
        reg_preds_test[target] = final_reg.predict(X_test)

    # ---- assemble submission ----
    pred_injured = (test_pred_cls >= 0.5).astype(int)
    lo, hi = train_df["onset_day_offset"].min(), train_df["onset_day_offset"].max()
    rlo, rhi = train_df["recovery_duration"].min(), train_df["recovery_duration"].max()
    onset = np.where(pred_injured == 1,
                      np.clip(np.round(reg_preds_test["onset_day_offset"]), lo, hi), 1).astype(int)
    recovery = np.where(pred_injured == 1,
                         np.clip(np.round(reg_preds_test["recovery_duration"]), rlo, rhi), 5).astype(int)

    submission = pd.DataFrame({
        "athlete_id": test_feat["Id"],
        "injured_in_risk_window": pred_injured,
        "onset_day_offset": onset,
        "recovery_duration": recovery,
    })
    submission.to_csv(ROOT / "submission.csv", index=False)

    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    # ---- plots for the deck ----
    plt.figure(figsize=(5, 4))
    y_cls.value_counts().sort_index().plot(kind="bar")
    plt.title("Class balance: injured_in_risk_window")
    plt.xlabel("label")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(REPORTS / "class_balance.png", dpi=120)
    plt.close()

    fi = pd.Series(final_clf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
    plt.figure(figsize=(6, 5))
    fi[::-1].plot(kind="barh")
    plt.title("Top 15 feature importances (classifier)")
    plt.tight_layout()
    plt.savefig(REPORTS / "feature_importance.png", dpi=120)
    plt.close()

    inj_rate = train_df.groupby("sport")["injured_in_risk_window"].mean().sort_values()
    plt.figure(figsize=(6, 4))
    inj_rate.plot(kind="barh")
    plt.title("Injury rate by sport")
    plt.tight_layout()
    plt.savefig(REPORTS / "injury_rate_by_sport.png", dpi=120)
    plt.close()

    return metrics


if __name__ == "__main__":
    train_and_predict()
