"""
=============================================================================
MODULE 7: INTELLIGENT RECOMMENDATION ENGINE
Project: AI Road Accident Prevention Intelligence System
=============================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Recommendation:
    category: str          # Speed | Fatigue | Weather | Distraction | …
    severity: str          # Warning | Alert | Critical
    icon: str
    headline: str
    detail: str
    action: str


PRIORITY = {'Critical': 0, 'Alert': 1, 'Warning': 2}


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(input_dict: dict,
                              adaptive_score: float = None) -> List[Recommendation]:
    """
    Generate context-aware prevention recommendations from a single observation.

    Parameters
    ----------
    input_dict     : raw feature dict (same keys as dataset)
    adaptive_score : pre-computed AdaptiveRiskScore (optional)

    Returns
    -------
    List[Recommendation] sorted by severity (Critical first)
    """
    recs: List[Recommendation] = []

    speed      = float(input_dict.get('Speed', 60))
    fatigue    = float(input_dict.get('Fatigue', 0))
    weather    = str(input_dict.get('Weather', 'Clear'))
    traffic    = float(input_dict.get('TrafficDensity', 5))
    road       = str(input_dict.get('RoadCondition', 'Good'))
    dist       = float(input_dict.get('Distraction', 0))
    vis        = float(input_dict.get('Visibility', 7))
    mobile     = float(input_dict.get('MobileUsage', 0))
    time_day   = str(input_dict.get('TimeOfDay', 'Morning'))
    vehicle    = str(input_dict.get('VehicleCondition', 'Good'))
    score      = adaptive_score if adaptive_score is not None else 5.0

    # ── SPEED ────────────────────────────────────────────────────────────────
    if speed > 140:
        recs.append(Recommendation(
            category='Speed', severity='Critical', icon='🚨',
            headline='EXTREME SPEED DETECTED',
            detail=f'Current speed {speed} km/h is dangerously above safe limits.',
            action='Reduce speed immediately to below 80 km/h. Pull over if needed.'
        ))
    elif speed > 100:
        recs.append(Recommendation(
            category='Speed', severity='Alert', icon='⚡',
            headline='High Speed Warning',
            detail=f'Speed {speed} km/h exceeds recommended limit.',
            action='Reduce to posted speed limit. Maintain 3-second following distance.'
        ))
    elif speed > 80:
        recs.append(Recommendation(
            category='Speed', severity='Warning', icon='⚠️',
            headline='Speed Advisory',
            detail=f'Speed {speed} km/h is slightly elevated.',
            action='Ease off accelerator. Be prepared for sudden stops.'
        ))

    # ── FATIGUE ──────────────────────────────────────────────────────────────
    if fatigue >= 8:
        recs.append(Recommendation(
            category='Fatigue', severity='Critical', icon='😴',
            headline='SEVERE FATIGUE — STOP NOW',
            detail=f'Fatigue level {fatigue}/10. Microsleep risk is extremely high.',
            action='Pull over at nearest rest stop immediately. Rest 20–30 minutes before continuing.'
        ))
    elif fatigue >= 6:
        recs.append(Recommendation(
            category='Fatigue', severity='Alert', icon='😪',
            headline='High Fatigue Alert',
            detail=f'Fatigue level {fatigue}/10. Reaction time is significantly impaired.',
            action='Take a 15-minute break, hydrate. Use highway rest area within next 10 km.'
        ))
    elif fatigue >= 4:
        recs.append(Recommendation(
            category='Fatigue', severity='Warning', icon='🥱',
            headline='Fatigue Warning',
            detail=f'Moderate fatigue ({fatigue}/10) detected.',
            action='Open windows, listen to alert music. Plan a rest break within 30 minutes.'
        ))

    # ── WEATHER ──────────────────────────────────────────────────────────────
    if weather in ('Stormy', 'Snowy'):
        recs.append(Recommendation(
            category='Weather', severity='Critical', icon='🌩️',
            headline=f'DANGEROUS {weather.upper()} CONDITIONS',
            detail='Extreme weather severely reduces traction and visibility.',
            action='Consider alternate route. If unavoidable — reduce speed by 50%, keep headlights on, maintain 6-second gap.'
        ))
    elif weather in ('Foggy', 'Rainy'):
        recs.append(Recommendation(
            category='Weather', severity='Alert', icon='🌧️',
            headline=f'{weather} Conditions Alert',
            detail=f'{weather} weather reduces stopping distance and visibility.',
            action='Reduce speed by 30%. Use fog/dipped headlights. Increase following distance to 4 seconds.'
        ))

    # ── ROAD CONDITION ───────────────────────────────────────────────────────
    if road in ('Icy', 'Damaged'):
        recs.append(Recommendation(
            category='Road', severity='Critical', icon='🛑',
            headline=f'{road.upper()} ROAD — EXTREME CAUTION',
            detail=f'{road} surface drastically increases skidding risk.',
            action='Reduce speed to 40 km/h. Avoid sudden braking. Use engine braking.'
        ))
    elif road in ('Wet', 'Under Construction'):
        recs.append(Recommendation(
            category='Road', severity='Alert', icon='🚧',
            headline=f'{road} Road Condition',
            detail='Reduced traction. Construction zones have unpredictable hazards.',
            action='Slow down. Follow construction zone speed limits. Watch for workers.'
        ))

    # ── DISTRACTION ──────────────────────────────────────────────────────────
    if dist >= 8 or mobile == 1:
        sev  = 'Critical' if dist >= 8 else 'Alert'
        recs.append(Recommendation(
            category='Distraction', severity=sev, icon='📵',
            headline='MOBILE / DISTRACTION ALERT',
            detail='Distracted driving multiplies accident probability by 400%.',
            action='Enable Do Not Disturb mode. Place phone in glove compartment. Eyes on road.'
        ))
    elif dist >= 5:
        recs.append(Recommendation(
            category='Distraction', severity='Warning', icon='👀',
            headline='Distraction Warning',
            detail=f'Distraction score {dist}/10 is elevated.',
            action='Focus attention ahead. Avoid adjusting GPS or music while driving.'
        ))

    # ── VISIBILITY ───────────────────────────────────────────────────────────
    if vis < 3:
        recs.append(Recommendation(
            category='Visibility', severity='Critical', icon='🌫️',
            headline='VERY LOW VISIBILITY',
            detail=f'Visibility {vis}/10 is critically low.',
            action='Switch on fog lights and hazards. Reduce speed to 30 km/h. Consider stopping safely.'
        ))
    elif vis < 5:
        recs.append(Recommendation(
            category='Visibility', severity='Alert', icon='🔦',
            headline='Poor Visibility Alert',
            detail=f'Visibility {vis}/10 is below safe threshold.',
            action='Use dipped headlights. Increase following distance. Do not overtake.'
        ))

    # ── NIGHT DRIVING ────────────────────────────────────────────────────────
    if time_day == 'Night':
        recs.append(Recommendation(
            category='TimeOfDay', severity='Warning', icon='🌙',
            headline='Night Driving Advisory',
            detail='Night driving carries 3× higher fatality risk.',
            action='Use full headlights. Watch for pedestrians. Reduce speed by 20%.'
        ))

    # ── TRAFFIC ──────────────────────────────────────────────────────────────
    if traffic >= 9:
        recs.append(Recommendation(
            category='Traffic', severity='Alert', icon='🚗',
            headline='Very High Traffic Density',
            detail=f'Traffic density {traffic}/10 — stop-and-go likely.',
            action='Maintain safe gap. Avoid lane changes. Stay alert for sudden braking.'
        ))

    # ── VEHICLE CONDITION ────────────────────────────────────────────────────
    if vehicle == 'Poor':
        recs.append(Recommendation(
            category='Vehicle', severity='Critical', icon='🔧',
            headline='POOR VEHICLE CONDITION',
            detail='Vehicle in poor state significantly increases mechanical failure risk.',
            action='Visit nearest service station. Check brakes, tyres, and lights before continuing.'
        ))

    # ── OVERALL SCORE ────────────────────────────────────────────────────────
    if score >= 8 and not recs:
        recs.append(Recommendation(
            category='General', severity='Critical', icon='🚨',
            headline='CRITICAL RISK LEVEL DETECTED',
            detail=f'Overall adaptive risk score: {score:.2f}/10.',
            action='Multiple risk factors combine to create extreme danger. Stop driving and assess situation.'
        ))

    # Sort by severity priority
    recs.sort(key=lambda r: PRIORITY.get(r.severity, 3))
    return recs


# ─────────────────────────────────────────────────────────────────────────────
# BATCH RECOMMENDATIONS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def get_recommendations_summary(input_dict: dict, adaptive_score: float = None) -> Dict:
    recs = generate_recommendations(input_dict, adaptive_score)
    criticals = [r for r in recs if r.severity == 'Critical']
    alerts    = [r for r in recs if r.severity == 'Alert']
    warnings  = [r for r in recs if r.severity == 'Warning']
    return {
        'total':       len(recs),
        'criticals':   len(criticals),
        'alerts':      len(alerts),
        'warnings':    len(warnings),
        'items':       recs,
        'top':         recs[0] if recs else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPER (plain text)
# ─────────────────────────────────────────────────────────────────────────────

def print_recommendations(recs: List[Recommendation]):
    print("\n" + "="*60)
    print("  INTELLIGENT PREVENTION RECOMMENDATIONS")
    print("="*60)
    if not recs:
        print("  ✅  No significant risks detected. Drive safely!")
        return
    for i, r in enumerate(recs, 1):
        print(f"\n  {i}. {r.icon}  [{r.severity}]  {r.headline}")
        print(f"     {r.detail}")
        print(f"     ➤  {r.action}")
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sample = {
        'Speed': 135, 'Fatigue': 7, 'Weather': 'Stormy',
        'TrafficDensity': 9, 'RoadCondition': 'Icy',
        'Distraction': 8, 'Visibility': 2.5,
        'TimeOfDay': 'Night', 'MobileUsage': 1,
        'VehicleCondition': 'Poor'
    }
    recs = generate_recommendations(sample, adaptive_score=9.2)
    print_recommendations(recs)
