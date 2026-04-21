"""
utils.py - Utility functions for Flight Delay Prediction System
"""

import pandas as pd
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

AIRLINES = ["AA", "DL", "UA", "SW", "WN", "B6", "AS", "NK", "F9", "G4"]

AIRPORTS = {
    "ATL": "Atlanta Hartsfield-Jackson",
    "LAX": "Los Angeles International",
    "ORD": "Chicago O'Hare",
    "DFW": "Dallas/Fort Worth",
    "JFK": "New York JFK",
    "MIA": "Miami International",
    "SFO": "San Francisco International",
    "DEN": "Denver International",
    "SEA": "Seattle-Tacoma",
    "LAS": "Las Vegas Harry Reid",
}

WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rainy", "Snowy", "Stormy", "Foggy"]

DAY_NAMES = {1: "Monday", 2: "Tuesday", 3: "Wednesday",
             4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}

WEATHER_DELAY_RISK = {
    "Clear": 0.10,
    "Cloudy": 0.20,
    "Rainy": 0.45,
    "Foggy": 0.40,
    "Snowy": 0.65,
    "Stormy": 0.80,
}

AIRLINE_NAMES = {
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "SW": "Southwest Airlines",
    "WN": "WN Airlines",
    "B6": "JetBlue Airways",
    "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
}

# ── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to raw dataframe."""
    df = df.copy()

    # Departure time → hour bucket (0-23)
    df["DepartureHour"] = df["ScheduledDeparture"] // 100

    # Rush-hour flag
    df["IsRushHour"] = df["DepartureHour"].apply(
        lambda h: 1 if (6 <= h <= 9) or (16 <= h <= 19) else 0
    )

    # Late-night flag
    df["IsLateNight"] = df["DepartureHour"].apply(
        lambda h: 1 if h >= 22 or h <= 5 else 0
    )

    # Weekend flag
    df["IsWeekend"] = df["DayOfWeek"].apply(lambda d: 1 if d >= 6 else 0)

    # Long-haul flag (>1500 miles)
    df["IsLongHaul"] = (df["Distance"] > 1500).astype(int)

    # Weather risk score
    df["WeatherRisk"] = df["WeatherCondition"].map(WEATHER_DELAY_RISK).fillna(0.1)

    return df


def preprocess_input(airline, origin, destination, dep_time,
                     distance, day_of_week, weather, encoders, scaler):
    """
    Convert raw user inputs into a model-ready feature vector.
    Returns a numpy array shaped (1, n_features).
    """
    dep_hour = dep_time // 100
    is_rush = 1 if (6 <= dep_hour <= 9) or (16 <= dep_hour <= 19) else 0
    is_late_night = 1 if dep_hour >= 22 or dep_hour <= 5 else 0
    is_weekend = 1 if day_of_week >= 6 else 0
    is_long_haul = 1 if distance > 1500 else 0
    weather_risk = WEATHER_DELAY_RISK.get(weather, 0.1)

    # Encode categoricals
    def safe_encode(encoder, value):
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        return 0  # unknown → 0

    airline_enc = safe_encode(encoders["Airline"], airline)
    origin_enc  = safe_encode(encoders["Origin"], origin)
    dest_enc    = safe_encode(encoders["Destination"], destination)
    weather_enc = safe_encode(encoders["WeatherCondition"], weather)

    features = np.array([[
        airline_enc, origin_enc, dest_enc,
        dep_time, distance, day_of_week,
        weather_enc, dep_hour, is_rush, is_late_night,
        is_weekend, is_long_haul, weather_risk
    ]])

    return scaler.transform(features)


def get_delay_interpretation(probability: float) -> dict:
    """Return a human-readable risk level and colour for a given probability."""
    if probability < 0.25:
        return {"level": "Low Risk", "color": "#22c55e", "emoji": "✅",
                "advice": "Your flight looks good! Minimal delay risk."}
    elif probability < 0.50:
        return {"level": "Moderate Risk", "color": "#f59e0b", "emoji": "⚠️",
                "advice": "Some delay possible. Arrive a bit early."}
    elif probability < 0.75:
        return {"level": "High Risk", "color": "#f97316", "emoji": "🔶",
                "advice": "Significant delay likely. Plan for extra waiting time."}
    else:
        return {"level": "Very High Risk", "color": "#ef4444", "emoji": "🚨",
                "advice": "Severe delay very likely. Consider rebooking or arriving early."}


def simulate_realtime_factors(airline, origin, weather, day_of_week, dep_hour):
    """
    Generate simulated real-time contributing factors for richer UI output.
    Returns a list of (factor_name, impact, direction) tuples.
    """
    factors = []

    weather_risk = WEATHER_DELAY_RISK.get(weather, 0.1)
    if weather_risk > 0.5:
        factors.append(("Weather Conditions", weather_risk, "negative"))
    elif weather_risk < 0.2:
        factors.append(("Weather Conditions", 1 - weather_risk, "positive"))

    if (6 <= dep_hour <= 9) or (16 <= dep_hour <= 19):
        factors.append(("Rush Hour Traffic", 0.6, "negative"))
    elif dep_hour >= 22 or dep_hour <= 5:
        factors.append(("Off-Peak Hours", 0.7, "positive"))
    else:
        factors.append(("Normal Hours", 0.5, "neutral"))

    if day_of_week in [5, 7]:   # Friday, Sunday
        factors.append(("High-Travel Day", 0.65, "negative"))
    elif day_of_week == 2:      # Tuesday
        factors.append(("Low-Travel Day", 0.75, "positive"))

    high_delay_airlines = ["NK", "F9", "G4"]
    low_delay_airlines  = ["AS", "DL"]
    if airline in high_delay_airlines:
        factors.append(("Airline On-Time Record", 0.4, "negative"))
    elif airline in low_delay_airlines:
        factors.append(("Airline On-Time Record", 0.8, "positive"))

    busy_airports = ["ORD", "ATL", "JFK", "LAX"]
    if origin in busy_airports:
        factors.append(("Origin Airport Congestion", 0.55, "negative"))

    return factors
