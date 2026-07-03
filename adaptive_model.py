"""
=============================================================================
MODULE 3: ADAPTIVE RISK SCORING ENGINE  (Patent Core Innovation)
Project: AI Road Accident Prevention Intelligence System

Formula:
  AccidentRisk = Ws*Speed + Wf*Fatigue + Ww*Weather + Wt*Traffic
               + Wr*RoadCondition + Wd*Distraction + Wv*Visibility

Weights self-adjust dynamically based on contextual rules.
=============================================================================
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# BASE WEIGHTS (sum = 1.0)
# ─────────────────────────────────────────────────────────────────────────────

BASE_WEIGHTS = {
    'Speed':       0.25,
    'Fatigue':     0.20,
    'Weather':     0.15,
    'Traffic':     0.15,
    'RoadCond':    0.10,
    'Distraction': 0.10,
    'Visibility':  0.05,
}

# Ordinal encodings for categorical columns
WEATHER_RISK = {
    'Clear': 1, 'Rainy': 3, 'Foggy': 4, 'Stormy': 5, 'Snowy': 5
}
ROAD_RISK = {
    'Good': 1, 'Wet': 3, 'Damaged': 4, 'Icy': 5, 'Under Construction': 4
}
TIME_RISK = {
    'Morning': 2, 'Afternoon': 2, 'Evening': 3, 'Night': 5
}
VEHICLE_RISK = {
    'Good': 1, 'Fair': 2, 'Poor': 4
}


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC WEIGHT OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

def compute_dynamic_weights(row: dict) -> dict:
    """
    Adjust base weights contextually so the most dangerous active factor
    receives amplified importance.

    Context rules
    ─────────────
    1. Severe weather       → boost Weather weight
    2. Night time           → boost Visibility weight
    3. High traffic density → boost Traffic weight
    4. High fatigue         → boost Fatigue weight
    5. High speed           → boost Speed weight
    6. Mobile usage active  → boost Distraction weight
    7. Poor road condition  → boost RoadCond weight
    """
    w = BASE_WEIGHTS.copy()

    weather   = str(row.get('Weather', 'Clear'))
    time_day  = str(row.get('TimeOfDay', 'Morning'))
    traffic   = float(row.get('TrafficDensity', 5))
    fatigue   = float(row.get('Fatigue', 0))
    speed     = float(row.get('Speed', 60))
    mobile    = float(row.get('MobileUsage', 0))
    road      = str(row.get('RoadCondition', 'Good'))

    boosts    = {}

    # Rule 1 — severe weather
    if weather in ('Stormy', 'Snowy', 'Foggy'):
        boosts['Weather'] = 0.10

    # Rule 2 — night visibility
    if time_day == 'Night':
        boosts['Visibility'] = 0.10

    # Rule 3 — high traffic
    if traffic >= 8:
        boosts['Traffic'] = 0.08

    # Rule 4 — high fatigue
    if fatigue >= 7:
        boosts['Fatigue'] = 0.08

    # Rule 5 — high speed
    if speed >= 120:
        boosts['Speed'] = 0.08

    # Rule 6 — mobile usage
    if mobile == 1:
        boosts['Distraction'] = 0.08

    # Rule 7 — poor road
    if road in ('Icy', 'Damaged', 'Under Construction'):
        boosts['RoadCond'] = 0.06

    # Apply boosts and re-normalise so weights still sum to 1.0
    total_boost = sum(boosts.values())
    if total_boost > 0:
        shrink = total_boost / len(w)
        for key in w:
            if key not in boosts:
                w[key] = max(0.01, w[key] - shrink)
        for key, val in boosts.items():
            w[key] += val

    # Normalise
    total = sum(w.values())
    w = {k: round(v / total, 4) for k, v in w.items()}
    return w


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE NORMALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm_speed(speed):       return np.clip(speed / 180, 0, 1) * 10
def _norm_fatigue(fatigue):   return np.clip(fatigue / 10, 0, 1) * 10
def _norm_traffic(traffic):   return np.clip(traffic / 10, 0, 1) * 10
def _norm_distraction(d):     return np.clip(d / 10, 0, 1) * 10
def _norm_visibility(vis):    return (1 - np.clip(vis / 10, 0, 1)) * 10   # inverse
def _norm_weather(w):         return WEATHER_RISK.get(str(w), 1) * 2
def _norm_road(r):            return ROAD_RISK.get(str(r), 1) * 2


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-ROW ADAPTIVE RISK SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_adaptive_risk_score(row: dict) -> dict:
    """
    Compute the Adaptive Risk Score for a single observation.

    Returns
    -------
    dict with keys:
        AdaptiveRiskScore  – float in [0, 10]
        RiskLabel          – Safe | Medium Risk | High Risk | Critical Risk
        Weights            – the dynamic weight dict used
        FeatureScores      – normalised individual factor scores
    """
    weights = compute_dynamic_weights(row)

    f_speed = _norm_speed(row.get('Speed', 60))
    f_fat   = _norm_fatigue(row.get('Fatigue', 0))
    f_wea   = _norm_weather(row.get('Weather', 'Clear'))
    f_traf  = _norm_traffic(row.get('TrafficDensity', 5))
    f_road  = _norm_road(row.get('RoadCondition', 'Good'))
    f_dist  = _norm_distraction(row.get('Distraction', 0))
    f_vis   = _norm_visibility(row.get('Visibility', 7))

    score = (
        weights['Speed']       * f_speed +
        weights['Fatigue']     * f_fat   +
        weights['Weather']     * f_wea   +
        weights['Traffic']     * f_traf  +
        weights['RoadCond']    * f_road  +
        weights['Distraction'] * f_dist  +
        weights['Visibility']  * f_vis
    )
    score = float(np.clip(score, 0, 10))

    if score < 3:
        label = 'Safe'
    elif score < 5:
        label = 'Medium Risk'
    elif score < 7:
        label = 'High Risk'
    else:
        label = 'Critical Risk'

    return {
        'AdaptiveRiskScore': round(score, 3),
        'RiskLabel':         label,
        'Weights':           weights,
        'FeatureScores': {
            'Speed':       round(f_speed, 2),
            'Fatigue':     round(f_fat,   2),
            'Weather':     round(f_wea,   2),
            'Traffic':     round(f_traf,  2),
            'RoadCond':    round(f_road,  2),
            'Distraction': round(f_dist,  2),
            'Visibility':  round(f_vis,   2),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# BATCH SCORING
# ─────────────────────────────────────────────────────────────────────────────

def apply_adaptive_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply compute_adaptive_risk_score to every row of a DataFrame.
    Adds columns: AdaptiveRiskScore, RiskLabel, DynamicWeightJSON
    """
    results = df.apply(lambda r: compute_adaptive_risk_score(r.to_dict()), axis=1)

    df['AdaptiveRiskScore'] = results.apply(lambda x: x['AdaptiveRiskScore'])
    df['RiskLabel']         = results.apply(lambda x: x['RiskLabel'])
    df['DynamicWeightJSON'] = results.apply(lambda x: str(x['Weights']))

    print(f"[✔] Adaptive scoring complete — {len(df):,} rows processed")
    print(f"    Risk distribution:\n{df['RiskLabel'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    sample = {
        'Speed': 145, 'Fatigue': 8, 'Weather': 'Stormy',
        'TrafficDensity': 9, 'RoadCondition': 'Icy',
        'Distraction': 7, 'Visibility': 2,
        'TimeOfDay': 'Night', 'MobileUsage': 1
    }
    result = compute_adaptive_risk_score(sample)
    print("\n=== Adaptive Risk Engine — Single Row Test ===")
    print(f"  Score  : {result['AdaptiveRiskScore']}")
    print(f"  Label  : {result['RiskLabel']}")
    print(f"  Weights: {result['Weights']}")
    print(f"  Factors: {result['FeatureScores']}")
