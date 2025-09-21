import math

import numpy as np
import pandas as pd
from scipy.stats import power
from sklearn.linear_model import LinearRegression
import datetime
from sklearn.linear_model import Ridge
from dataclasses import dataclass

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 ------ Builds a data model that calculates and holds the coefficients for the athlete ------
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

@dataclass
class AthleteModel:
    # Coefficients for: intercept, logd, logd^2, heat_above60, hum_above60, elev_gain
    coef_: np.ndarray  # shape (6,)
    features: list      # feature names for clarity

def build_design_matrix(dist_m_list, temp_f_list, hum_list, elev_m_list, surface_list=None):
    dist_m = np.asarray(dist_m_list, dtype=float)
    logd = np.log(dist_m)
    heat_above60 = np.maximum(0.0, np.asarray(temp_f_list, dtype=float) - 60.0)
    hum_above60 = np.maximum(0.0, np.asarray(hum_list, dtype=float) - 60.0)
    elev_gain = np.maximum(0.0, np.asarray(elev_m_list, dtype=float))

    # Surface encoding: track=0, grass=1, trail=2
    grass_flag = []
    trail_flag = []
    if surface_list is not None:
        for s in surface_list:
            s = (s or "").strip().lower()
            grass_flag.append(1.0 if s == "grass" else 0.0)
            trail_flag.append(1.0 if s == "trail" else 0.0)
    else:
        grass_flag = [0.0] * len(dist_m)
        trail_flag = [0.0] * len(dist_m)

    X = np.column_stack([
        np.ones_like(logd),   # intercept
        logd,
        logd**2,
        heat_above60,
        hum_above60,
        elev_gain,
        grass_flag,
        trail_flag
    ])
    feature_names = ["intercept", "logd", "logd2", "heat_above60", "hum_above60", "elev_gain", "grass", "trail"]
    return X, feature_names


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 ------ Utility functions for data parsing & conversion   -------
 - parse_time_to_seconds: convert a "M:SS" or "M:SS.ss" string into total seconds (float)
 - seconds_to_time: format a float number of seconds back into "M:SS.ss" string
 - to_int_safe: safely convert a value to int (returns default if it fails)
 - to_float_safe: safely convert a value to float (returns default if it fails)
 - to_date_safe: parse a date string ("M/D/YYYY") into a datetime.date object

 These functions keep input handling robust by catching bad/missing values,
 ensuring the rest of the modeling code always works with clean, numeric data.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def parse_time_to_seconds(time_str):
    """
    Accepts formats like '12:34.56' or '12:34' and returns total seconds as float.
    """
    if not time_str or ":" not in time_str:
        raise ValueError(f"Invalid time string: {time_str}")
    m_str, s_str = time_str.strip().split(":")
    return int(m_str) * 60 + float(s_str)

def seconds_to_time(seconds):
    """
    Converts seconds (float) to 'M:SS.ss' format.
    """
    if seconds is None or np.isnan(seconds):
        return None
    seconds = float(seconds)
    minutes = int(seconds // 60)
    remaining_seconds = seconds - minutes * 60
    return f"{minutes}:{remaining_seconds:05.2f}"  # zero-pad seconds to at least 2 digits

def to_int_safe(v, default=None):
    try:
        return int(str(v).strip())
    except Exception:
        return default

def to_float_safe(v, default=None):
    try:
        return float(str(v).strip())
    except Exception:
        return default

def to_date_safe(date_str, default=None):
    """
    Expects M/D/YYYY or MM/DD/YYYY
    """
    try:
        month, day, year = map(int, date_str.strip().split("/"))
        return datetime.date(year, month, day)
    except Exception:
        return default


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
  ------ Time predicting functions -------
  - The data model of the athlete created are used to predict the time of the runner in ideal, perfect conditions
  - The Gascon Model is a model for predicting race times across different distances
  - A heuristic approach is taken if their is an insufficient amount of data to use the default model
 """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def predict_time_with_conditions(ath_model, distance_m, temp_f, humidity, elev_gain, surface=None):
    """
    Predict time given distance and conditions using the athlete model.
    """
    if ath_model is None or distance_m is None or distance_m <= 0:
        return None
    # Pass surface to design matrix
    X, names = build_design_matrix([distance_m], [temp_f], [humidity], [elev_gain], surface_list=[surface])
    y_pred = float(np.dot(X[0], ath_model.coef_))
    return y_pred

def learned_normalize_time(ath_model, time_sec, distance_m, temp_f, humidity, elev_gain, surface=None):
    """
    Convert an observed time to an athlete-specific 'ideal' time using learned condition effects,
    now including surface effect.
    """
    if ath_model is None:
        return normalize_time(time_sec, distance_m, temp_f, humidity, elev_gain)

    pred_actual = predict_time_with_conditions(ath_model, distance_m, temp_f, humidity, elev_gain, surface=surface)
    pred_ideal = predict_time_with_conditions(ath_model, distance_m, 60.0, 60.0, 0.0, surface="track")

    if pred_actual is None or pred_ideal is None or pred_actual <= 0:
        return normalize_time(time_sec, distance_m, temp_f, humidity, elev_gain)

    # Scale observed performance by ratio of ideal to actual predictions
    return float(time_sec) * (pred_ideal / pred_actual)


def fit_athlete_condition_model(runners, athlete, min_samples=4, alpha=1.0):
    """
    Fit an athlete-specific Ridge regression to learn condition effects.
    Returns AthleteModel or None if insufficient data.
    """
    dist, temp, hum, elev, times = [], [], [], [], []

    for r in runners:
        if r.get("Athlete", "").strip() != athlete.strip():
            continue
        d = to_float_safe(r.get("Distance (m)"))
        t = to_float_safe(r.get("Temperature (F)"))
        h = to_float_safe(r.get("Humidity (%)"))
        e = to_float_safe(r.get("Elevation Gain"))
        time_sec = None
        try:
            time_sec = parse_time_to_seconds(r.get("Time", ""))
        except Exception:
            pass
        if d and d > 0 and t is not None and h is not None and e is not None and time_sec is not None:
            dist.append(d); temp.append(t); hum.append(h); elev.append(e); times.append(time_sec)

    if len(times) < min_samples:
        return None

    X, feature_names = build_design_matrix(dist, temp, hum, elev)
    y = np.asarray(times, dtype=float)

    # Ridge for stability; you can tune alpha
    model = Ridge(alpha=alpha, fit_intercept=False).fit(X, y)

    coef = model.coef_.copy()
    heat_idx = feature_names.index("heat_above60")
    hum_idx = feature_names.index("hum_above60")
    elev_idx = feature_names.index("elev_gain")
    coef[heat_idx] = max(coef[heat_idx], 0.0)
    coef[hum_idx] = max(coef[hum_idx], 0.0)
    coef[elev_idx] = max(coef[elev_idx], 0.0)

    return AthleteModel(coef_=coef, features=feature_names)

runners_data = []  # global for demo purposes

def vo2(distance: int, time: int):
    """
    Returns (VO2_estimate_ml_per_kg_min, percent_vo2, VDOT)
    distance_m: meters (e.g., 5000)
    time_s: seconds (e.g., 20*60 for 20 minutes)
    Based on Daniels & Gilbert VDOT formula (speed converted to m/min).
    """
    # speed
    v_mps = distance / time              # m/s
    v_m_per_min = v_mps * 60.0                 # m/min

    # estimated VO2 cost of that speed (ml·kg^-1·min^-1)
    vo2 = 0.182258 * v_m_per_min + 0.000104 * (v_m_per_min**2) + 3.5

    # percent of VO2max sustainable for race duration T (T in minutes)
    t_min = time / 60.0
    percent = (0.8
               + 0.1894393 * math.exp(-0.012778 * t_min)
               + 0.2989558 * math.exp(-0.1932605 * t_min))

    # VDOT estimate
    vdot = vo2 / percent

    return round(vo2, 2)

def normalize_time(time_sec, distance_m, temp_f, humidity, elevation_gain_m, athlete=None, runners=None):
    if athlete and runners:
        ath_model = fit_athlete_condition_model(runners, athlete)
        if ath_model:
            return learned_normalize_time(ath_model, time_sec, distance_m, temp_f, humidity, elevation_gain_m)

    heuristic(time_sec, distance_m, temp_f, humidity, elevation_gain_m, athlete=athlete, runners=runners)

def predict_time(model, distance_m):
    distance_m = float(distance_m)
    if model is None or distance_m <= 0:
        return None
    x = np.array([[np.log(distance_m), np.log(distance_m) ** 2]])
    y_pred = model.predict(x)
    return float(y_pred[0]) if y_pred is not None else None

def heuristic(time_sec, distance_m, temp_f, humidity, elevation_gain_m, athlete=None, runners=None):

    temp_penalty = max(0.0, temp_f - 60.0) * 0.003

    # Humidity penalty: 0.15% per % above 60
    hum_penalty = max(0.0, humidity - 60.0) * 0.0015

    # Elevation penalty: 0.01 × (gain in meters / sqrt(distance in meters))
    # This reduces effect on short races while still penalizing big hills
    elev_penalty = 0.01 * max(0.0, elevation_gain_m) / np.sqrt(max(1.0, distance_m))

    factor = 1.0 + temp_penalty + hum_penalty + elev_penalty
    return time_sec / factor


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
  ------ Averaging functions -------
  - These functions calculate different averages according to the race times
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def average_ideal_for_distance(runners, athlete, distance):
    distance = to_int_safe(distance, None)
    if distance is None:
        return None
    times = []
    for r in runners:
        try:
            if r.get("Athlete", "").strip() == athlete.strip() and to_int_safe(r.get("Distance (m)")) == distance:
                total_sec = parse_time_to_seconds(r.get("Time", ""))
                temp = to_float_safe(r.get("Temperature (F)"), 60)
                hum = to_float_safe(r.get("Humidity (%)"), 60)
                elev = to_float_safe(r.get("Elevation Gain"), 0)
                normal = normalize_time(total_sec, distance, temp, hum, elev, athlete=athlete, runners=runners)
                if normal is not None:
                    times.append(normal)
        except Exception:
            continue
    if not times:
        return None
    return float(np.mean(times))

def average_time_for_distance(runners, athlete, distance):
    distance = to_int_safe(distance, None)
    if distance is None:
        return None

    times = []
    for r in runners:
        if r.get("Athlete", "").strip() == athlete.strip() and to_int_safe(r.get("Distance (m)")) == distance:
            try:
                total_sec = parse_time_to_seconds(r.get("Time", ""))
                times.append(total_sec)
            except Exception:
                continue
    if not times:
        return None
    return float(np.mean(times))

def cumulative_averages(runners, athlete, distance):
    distance = to_int_safe(distance, None)
    if distance is None:
        return [], [], []

    raw_times = []
    ideal_times = []
    raw_averages = []
    ideal_averages = []
    events = []

    # Sort by date to make cumulative meaningful
    rows = [
        (to_date_safe(r.get("Date", "")),r) for r in runners if r.get("Athlete", "").strip() == athlete.strip()
        and to_int_safe(r.get("Distance (m)")) == distance
    ]
    rows.sort(key=lambda x: (x[0] is None, x[0]))  # None dates last, but included

    for _, r in rows:
        try:
            total_sec = parse_time_to_seconds(r.get("Time", ""))
            temp = to_float_safe(r.get("Temperature (F)"), 60)
            hum = to_float_safe(r.get("Humidity (%)"), 60)
            elev = to_float_safe(r.get("Elevation Gain"), 0)
            normal = normalize_time(total_sec, distance, temp, hum, elev, athlete=athlete, runners=runners)

            if total_sec is None or normal is None:
                continue

            raw_times.append(total_sec)
            ideal_times.append(normal)
            raw_averages.append(float(np.mean(raw_times)))
            ideal_averages.append(float(np.mean(ideal_times)))
            events.append(r.get("Event", "").strip())
        except Exception:
            continue

    return raw_averages, ideal_averages, events

def fit_gascon_model(distances_m, times_sec):
    """
    Simple quadratic in log(distance) as per Gascon-like model variant.
    """
    distances_m = np.asarray(distances_m, dtype=float)
    times_sec = np.asarray(times_sec, dtype=float)

    # Filter out nonpositive distances and missing times
    mask = (distances_m > 0) & np.isfinite(times_sec)
    distances_m = distances_m[mask]
    times_sec = times_sec[mask]

    if len(distances_m) < 2:
        return None

    X = np.column_stack([np.log(distances_m), np.log(distances_m) ** 2])
    y = times_sec
    model = LinearRegression().fit(X, y)
    return model

def weighted(runners, athlete, distance):
    distance = to_int_safe(distance, None)
    if distance is None:
        return []

    events, times, dates = [], [], []
    for r in runners:
        try:
            if r.get("Athlete", "").strip() == athlete.strip() and to_int_safe(r.get("Distance (m)")) == distance:
                total_sec = parse_time_to_seconds(r.get("Time", ""))
                date = to_date_safe(r.get("Date", ""))
                # Skip rows without valid date or time
                if date is None or total_sec is None:
                    continue
                times.append(total_sec)
                dates.append(date)
                events.append(r.get("Event", "").strip())
        except Exception:
            continue

    if not times:
        return []

    df = pd.DataFrame({"Race": events, "Time (s)": times, "Date": dates})
    df = df.sort_values("Date")
    today = df["Date"].max()
    df["WeeksAgo"] = df["Date"].apply(lambda d: (today - d).days / 7.0)
    lambda_val = 0.05  # decay rate per week
    df["Weight"] = np.exp(-lambda_val * df["WeeksAgo"])

    weighted_series = []
    for i in range(1, len(df) + 1):
        weighted_avg = np.average(df["Time (s)"].iloc[:i], weights=df["Weight"].iloc[:i])
        weighted_series.append(float(weighted_avg))

    return weighted_series
