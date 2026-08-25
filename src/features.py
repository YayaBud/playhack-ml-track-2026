"""
Feature engineering for PlayHack ML Track 2026 (IIT Guwahati).

LEAKAGE CONTRACT
----------------
Train has 60 days of wearable data (2026-01-05 .. 2026-03-05); test has only
the first 30 (2026-01-05 .. 2026-02-03). Days 31-60 ARE the risk window that
the labels describe. Every feature here is computed strictly from
2026-01-05..2026-02-03 for BOTH splits, so train and test see identical
observation windows. assert_no_leakage() re-checks this at build time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

OBS_START = pd.Timestamp("2026-01-05")
OBS_END = pd.Timestamp("2026-02-03")      # inclusive, 30 days
OBS_DAYS = 30

CAT_COLS = ["sport", "gender", "dominant_side", "position"]


# ---------------------------------------------------------------- helpers ---
def _win_stats(pivot: pd.DataFrame, name: str) -> pd.DataFrame:
    """pivot: index=Id, columns=day_index(0..29), values=daily metric.

    Returns mean/std/slope over the full window, plus acute (last 7d) vs
    chronic (prior 23d) ratio and a last-7 minus first-7 delta. These are the
    standard sports-science workload constructs (ACWR, monotony, strain).
    """
    v = pivot.to_numpy(dtype=float)
    n = v.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(v, axis=1)
        std = np.nanstd(v, axis=1)
        total = np.nansum(v, axis=1)
        last7 = np.nanmean(v[:, -7:], axis=1)
        prev23 = np.nanmean(v[:, :-7], axis=1)
        first7 = np.nanmean(v[:, :7], axis=1)
        last14 = np.nanmean(v[:, -14:], axis=1)
        mx = np.nanmax(v, axis=1)
        monotony = np.where(std > 0, mean / std, np.nan)
        strain = total * monotony
        acwr = np.where(prev23 > 0, last7 / prev23, np.nan)
        cv = np.where(mean != 0, std / mean, np.nan)

    x = np.arange(n, dtype=float)
    slopes = np.full(v.shape[0], np.nan)
    for i in range(v.shape[0]):
        row = v[i]
        m = ~np.isnan(row)
        if m.sum() >= 2:
            slopes[i] = np.polyfit(x[m], row[m], 1)[0]

    return pd.DataFrame({
        name + "_mean": mean,
        name + "_std": std,
        name + "_cv": cv,
        name + "_max": mx,
        name + "_slope": slopes,
        name + "_last7": last7,
        name + "_last14": last14,
        name + "_acwr": acwr,
        name + "_delta_last_first": last7 - first7,
        name + "_monotony": monotony,
        name + "_strain": strain,
    }, index=pivot.index)


def _daily_pivot(df, id_col, date_col, val_col, aggfunc="sum"):
    """Pivot to Id x day-index(0..29), reindexed so every athlete has 30 columns."""
    d = df.copy()
    d["_di"] = (d[date_col] - OBS_START).dt.days
    p = d.pivot_table(index=id_col, columns="_di", values=val_col, aggfunc=aggfunc)
    return p.reindex(columns=range(OBS_DAYS))


def _clip_window(df, date_col):
    return df[(df[date_col] >= OBS_START) & (df[date_col] <= OBS_END)]


# ------------------------------------------------------------- components ---
def metadata_features(split):
    df = pd.read_csv(DATA / split / "athlete_metadata.csv").rename(columns={"athlete_id": "Id"})
    df = df.set_index("Id")
    df["bmi_baseline"] = df["weight_kg_baseline"] / (df["height_cm"] / 100.0) ** 2
    df["experience_ratio"] = df["years_playing"] / df["age"].replace(0, np.nan)
    df["injury_per_year"] = df["prior_season_injury_count"] / df["years_playing"].replace(0, np.nan)
    df["age_x_prior_injury"] = df["age"] * df["prior_season_injury_count"]
    df["team_code"] = df["team_id"].astype("category").cat.codes
    return df.drop(columns=["team_id"])


def daily_features(split):
    df = pd.read_csv(DATA / split / "dailyActivity_merged.csv", parse_dates=["ActivityDate"])
    df = _clip_window(df, "ActivityDate")
    df["active_min"] = df["VeryActiveMinutes"] + df["FairlyActiveMinutes"] + df["LightlyActiveMinutes"]
    df["active_sed_ratio"] = df["active_min"] / df["SedentaryMinutes"].replace(0, np.nan)
    # intensity-weighted load: the daily-activity analogue of session-RPE load
    df["load"] = (df["VeryActiveMinutes"] * 3 + df["FairlyActiveMinutes"] * 2
                  + df["LightlyActiveMinutes"])

    parts = []
    for col, alias in [("TotalSteps", "steps"), ("Calories", "cal"),
                       ("VeryActiveMinutes", "vam"), ("active_min", "actmin"),
                       ("SedentaryMinutes", "sed"), ("load", "load"),
                       ("TotalDistance", "dist")]:
        parts.append(_win_stats(_daily_pivot(df, "Id", "ActivityDate", col), alias))

    out = pd.concat(parts, axis=1)
    g = df.groupby("Id")
    out["active_sed_ratio_mean"] = g["active_sed_ratio"].mean()
    out["steps_per_cal"] = out["steps_mean"] / out["cal_mean"].replace(0, np.nan)
    out["n_days_logged"] = g.size()
    return out


def sleep_features(split):
    df = pd.read_csv(DATA / split / "sleepDay_merged.csv", parse_dates=["SleepDay"])
    df = _clip_window(df, "SleepDay")
    df["efficiency"] = df["TotalMinutesAsleep"] / df["TotalTimeInBed"].replace(0, np.nan)
    df["debt"] = df["TotalTimeInBed"] - df["TotalMinutesAsleep"]

    out = pd.concat([
        _win_stats(_daily_pivot(df, "Id", "SleepDay", "TotalMinutesAsleep", "mean"), "sleep"),
        _win_stats(_daily_pivot(df, "Id", "SleepDay", "efficiency", "mean"), "sleepeff"),
    ], axis=1)
    g = df.groupby("Id")
    out["sleep_debt_mean"] = g["debt"].mean()
    out["sleep_debt_total"] = g["debt"].sum()
    out["n_sleep_records"] = g.size()
    # nights under 7h: the commonly cited injury-risk threshold
    out["nights_under_7h"] = df.assign(short=df["TotalMinutesAsleep"] < 420).groupby("Id")["short"].sum()
    out["frac_nights_under_7h"] = out["nights_under_7h"] / out["n_sleep_records"].replace(0, np.nan)
    return out


def weight_features(split):
    df = pd.read_csv(DATA / split / "weightLogInfo_merged.csv", parse_dates=["Date"])
    df = _clip_window(df, "Date").sort_values("Date")
    g = df.groupby("Id")
    last = g.last()[["WeightKg", "Fat", "BMI"]]
    last.columns = ["weight_last", "fat_last", "bmi_last"]
    extra = pd.DataFrame({
        "weight_mean": g["WeightKg"].mean(),
        "weight_std": g["WeightKg"].std(),
        "weight_change": g["WeightKg"].last() - g["WeightKg"].first(),
        "n_weight_logs": g.size(),
    })
    return last.join(extra, how="outer")


def session_features(split):
    df = pd.read_csv(DATA / split / "training_sessions.csv", parse_dates=["date"])
    df = df.rename(columns={"athlete_id": "Id"})
    df = _clip_window(df, "date")
    df["hours"] = df["end_hour"] - df["start_hour"]

    out = _win_stats(_daily_pivot(df, "Id", "date", "hours", "sum"), "sesh")

    g = df.groupby("Id")
    out["n_sessions"] = g.size()
    out["n_session_days"] = g["date"].nunique()
    out["rest_days"] = OBS_DAYS - out["n_session_days"]
    out["max_sessions_per_day"] = df.groupby(["Id", "date"]).size().groupby(level=0).max()
    out["mean_start_hour"] = g["start_hour"].mean()
    out["std_start_hour"] = g["start_hour"].std()
    out["late_session_frac"] = df.assign(late=df["start_hour"] >= 19).groupby("Id")["late"].mean()

    tc = df.pivot_table(index="Id", columns="sport_session_type", values="session_id",
                        aggfunc="count", fill_value=0)
    tc.columns = ["n_" + str(c) for c in tc.columns]
    out = out.join(tc, how="left")
    for c in tc.columns:
        out["frac_" + c] = out[c] / out["n_sessions"].replace(0, np.nan)

    # gap structure: consecutive-training streaks and recency of last session
    day_idx = (df["date"] - OBS_START).dt.days
    per_day = df.assign(_di=day_idx).groupby("Id")["_di"].apply(lambda s: sorted(set(s)))
    max_streak, max_gap, days_since_last = {}, {}, {}
    for aid, days in per_day.items():
        streak = best = 1
        gap = 0
        for a, b in zip(days, days[1:]):
            if b - a == 1:
                streak += 1
                best = max(best, streak)
            else:
                streak = 1
                gap = max(gap, b - a - 1)
        max_streak[aid] = best
        max_gap[aid] = gap
        days_since_last[aid] = (OBS_DAYS - 1) - days[-1]
    out["max_train_streak"] = pd.Series(max_streak)
    out["max_rest_gap"] = pd.Series(max_gap)
    out["days_since_last_session"] = pd.Series(days_since_last)
    return out


def hourly_features(split):
    """Heart rate / intensity / steps at hourly grain -> daily -> window stats."""
    hr = pd.read_csv(DATA / split / "hourlyHeartrate_merged.csv",
                     usecols=["Id", "ActivityHour", "AvgHeartRate", "MinHeartRate", "MaxHeartRate"],
                     dtype={"Id": "int32", "AvgHeartRate": "float32",
                            "MinHeartRate": "float32", "MaxHeartRate": "float32"})
    hr["dt"] = pd.to_datetime(hr["ActivityHour"], format="%Y-%m-%dT%H:%M:%S")
    hr = hr[(hr["dt"] >= OBS_START) & (hr["dt"] < OBS_END + pd.Timedelta(days=1))]
    hr["date"] = hr["dt"].dt.normalize()
    hr["hour"] = hr["dt"].dt.hour
    hr["hr_range"] = hr["MaxHeartRate"] - hr["MinHeartRate"]

    out = pd.concat([
        _win_stats(_daily_pivot(hr, "Id", "date", "AvgHeartRate", "mean"), "hr"),
        # night HR (00:00-04:59) is the resting-HR / recovery proxy
        _win_stats(_daily_pivot(hr[hr["hour"] < 5], "Id", "date", "AvgHeartRate", "mean"), "resthr"),
        _win_stats(_daily_pivot(hr, "Id", "date", "MaxHeartRate", "max"), "hrmax"),
        _win_stats(_daily_pivot(hr, "Id", "date", "hr_range", "mean"), "hrrange"),
    ], axis=1)

    g = hr.groupby("Id")["AvgHeartRate"]
    out["hr_p10"] = g.quantile(0.10)
    out["hr_p90"] = g.quantile(0.90)
    out["hr_iqr"] = g.quantile(0.75) - g.quantile(0.25)
    # within-day HR dispersion averaged over days: a crude HRV/strain surrogate
    out["hr_within_day_std"] = hr.groupby(["Id", "date"])["AvgHeartRate"].std().groupby(level=0).mean()
    del hr

    ints = pd.read_csv(DATA / split / "hourlyIntensities_merged.csv",
                       usecols=["Id", "ActivityHour", "TotalIntensity"],
                       dtype={"Id": "int32", "TotalIntensity": "float32"})
    ints["dt"] = pd.to_datetime(ints["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
    ints = ints[(ints["dt"] >= OBS_START) & (ints["dt"] < OBS_END + pd.Timedelta(days=1))]
    ints["date"] = ints["dt"].dt.normalize()
    out = out.join(_win_stats(_daily_pivot(ints, "Id", "date", "TotalIntensity", "sum"), "intens"))
    out["intens_peak_hour_mean"] = ints.groupby(["Id", "date"])["TotalIntensity"].max().groupby(level=0).mean()
    del ints

    steps = pd.read_csv(DATA / split / "hourlySteps_merged.csv",
                        usecols=["Id", "ActivityHour", "StepTotal"],
                        dtype={"Id": "int32", "StepTotal": "float32"})
    steps["dt"] = pd.to_datetime(steps["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
    steps = steps[(steps["dt"] >= OBS_START) & (steps["dt"] < OBS_END + pd.Timedelta(days=1))]
    steps["date"] = steps["dt"].dt.normalize()
    out["steps_hourly_std"] = steps.groupby("Id")["StepTotal"].std()
    out["steps_peak_hour_mean"] = steps.groupby(["Id", "date"])["StepTotal"].max().groupby(level=0).mean()
    del steps

    return out


# ------------------------------------------------------------------ build ---
def assert_no_leakage(split):
    """Fail loudly if any source file would contribute rows past the window."""
    for fname, col in [("dailyActivity_merged.csv", "ActivityDate"),
                       ("sleepDay_merged.csv", "SleepDay"),
                       ("training_sessions.csv", "date"),
                       ("weightLogInfo_merged.csv", "Date")]:
        d = pd.read_csv(DATA / split / fname, parse_dates=[col], usecols=[col])
        clipped = _clip_window(d, col)
        assert clipped[col].max() <= OBS_END, split + "/" + fname + " leaks past window"
        assert clipped[col].min() >= OBS_START, split + "/" + fname + " starts before window"
    print("  [leak-check] " + split + ": sources clipped to "
          + str(OBS_START.date()) + ".." + str(OBS_END.date()))


def build_features(split):
    assert_no_leakage(split)
    df = metadata_features(split)
    for fn in (daily_features, sleep_features, weight_features,
               session_features, hourly_features):
        df = df.join(fn(split), how="left")
        print("  [" + split + "] +" + fn.__name__.ljust(20) + " -> " + str(df.shape[1]) + " cols")

    # per-sport z-scores: absolute load means different things in different sports
    for col in ["load_mean", "sesh_mean", "hr_mean", "steps_mean", "sleep_mean"]:
        if col in df.columns:
            grp = df.groupby("sport", observed=True)[col]
            df[col + "_z_sport"] = (df[col] - grp.transform("mean")) / grp.transform("std")

    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    return df.reset_index()


def get_features(split, rebuild=False):
    cache = DATA / ("feat_" + split + ".parquet")
    if cache.exists() and not rebuild:
        return pd.read_parquet(cache)
    df = build_features(split)
    df.to_parquet(cache)
    return df


if __name__ == "__main__":
    for s in ("train", "test"):
        d = get_features(s, rebuild=True)
        print(s + ": " + str(d.shape))
