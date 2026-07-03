"""
=============================================================================
MODULE 6: EXPLAINABLE AI ENGINE (SHAP)
Project: AI Road Accident Prevention Intelligence System
=============================================================================
"""

import numpy as np
import pandas as pd
import shap
import joblib
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BG = '#0a0e1a'; ACCENT = '#00d4ff'; FONT = '#e0e8ff'
_L = dict(paper_bgcolor=BG, plot_bgcolor=BG,
          font=dict(color=FONT, family='Orbitron, monospace'),
          title_font=dict(size=16, color=ACCENT))


# ─────────────────────────────────────────────────────────────────────────────
# SHAP EXPLAINER — build once and cache
# ─────────────────────────────────────────────────────────────────────────────

_explainer_cache = {}


def get_explainer(model, X_background: np.ndarray, model_type: str = 'tree'):
    """
    Build and cache a SHAP explainer.
    Uses TreeExplainer for forest/boosting models, LinearExplainer otherwise.
    """
    global _explainer_cache
    key = id(model)
    if key in _explainer_cache:
        return _explainer_cache[key]

    bg = shap.sample(X_background, min(200, len(X_background)))
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model, bg)
    else:
        explainer = shap.LinearExplainer(model, bg)

    _explainer_cache[key] = explainer
    return explainer

def compute_shap_values(model, X, feature_names,
                        model_type='tree', n_sample=500):

    idx = np.random.choice(len(X), min(n_sample, len(X)), replace=False)
    X_s = X[idx]

    explainer = get_explainer(model, X, model_type)

    shap_vals = explainer(X_s)

    # Latest SHAP API
    if hasattr(shap_vals, "values"):
        shap_arr = shap_vals.values
    else:
        shap_arr = shap_vals

    shap_arr = np.array(shap_arr)

    # Binary classification
    if shap_arr.ndim == 2:
        pass

    # Multi-class (samples, features, classes)
    elif shap_arr.ndim == 3:
        shap_arr = np.mean(np.abs(shap_arr), axis=2)

    # Old SHAP format
    elif isinstance(shap_vals, list):
        shap_arr = np.mean(np.abs(np.array(shap_vals)), axis=0)

    return shap_arr, explainer, X_s

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(shap_arr, feature_names):

    shap_arr = np.asarray(shap_arr)

    if shap_arr.ndim == 3:
        shap_arr = np.mean(np.abs(shap_arr), axis=2)

    mean_abs = np.mean(np.abs(shap_arr), axis=0)

    order = np.argsort(mean_abs)

    fig = go.Figure(go.Bar(
        x=mean_abs[order],
        y=np.array(feature_names)[order],
        orientation='h',
        marker=dict(
            color=mean_abs[order],
            colorscale='Plasma'
        )
    ))

    fig.update_layout(
        **_L,
        title="🧩 SHAP Feature Importance",
        xaxis_title="Mean |SHAP|",
        yaxis_title="Features"
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE PREDICTION EXPLANATION
# ─────────────────────────────────────────────────────────────────────────────

def explain_single_prediction(model, scaler, feature_names: list,
                               input_dict: dict, model_type: str = 'tree'):
    """
    Generate a SHAP waterfall explanation for ONE input observation.
    Returns (plotly figure, text_report).
    """
    row = pd.DataFrame([input_dict])
    # Ensure encoded columns exist
    from sklearn.preprocessing import LabelEncoder
    for col in ['Weather', 'RoadCondition', 'TimeOfDay', 'VehicleCondition', 'Country']:
        enc_col = col + '_enc'
        if enc_col in feature_names and col in row.columns:
            le = LabelEncoder()
            row[enc_col] = le.fit_transform(row[col].astype(str))

    # Ensure every feature exists
    for feature in feature_names:
        if feature not in row.columns:
             row[feature] = 0

    X_row = row[feature_names].fillna(0).values
    X_row = scaler.transform(X_row)

    explainer = get_explainer(model, X_row, model_type)
    sv = explainer(X_row)

    if hasattr(sv, "values"):
      sv = sv.values

      sv = np.array(sv)

    if sv.ndim == 3:
      sv_row = np.mean(np.abs(sv), axis=2)[0]
    elif sv.ndim == 2:
      sv_row = sv[0]
    else:
      sv_row = np.array(sv).flatten()



    # Build bar chart
    # Ensure SHAP values are 1-D
    sv_row = np.asarray(sv_row).flatten()

# Match feature count
    n = min(len(feature_names), len(sv_row))

    pairs = []

    for i in range(n):
       pairs.append(
        (feature_names[i], float(sv_row[i]))
    )

    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    pairs = pairs[:10]
    feats_  = [p[0] for p in pairs]
    vals_   = [p[1] for p in pairs]
    colours = ['#ff6b6b' if v > 0 else '#2ecc71' for v in vals_]

    fig = go.Figure(go.Bar(
        x=vals_, y=feats_,
        orientation='h',
        marker_color=colours,
    ))
    fig.update_layout(**_L,
                      title='🔍 Single Prediction — SHAP Explanation',
                      xaxis_title='SHAP Contribution',
                      yaxis=dict(autorange='reversed'))

    # Text report
    report_lines = ["EXPLAINABLE AI REPORT", "=" * 40]
    for f, v in pairs:
        direction = '▲ INCREASES' if v > 0 else '▼ DECREASES'
        report_lines.append(f"  {f:25s}  {direction} risk  ({v:+.4f})")

    top = pairs[0]
    report_lines += [
        "",
        f"PRIMARY DRIVER  : {top[0]}",
        f"CONTRIBUTION    : {top[1]:+.4f}",
        "",
        "The model flagged this scenario as HIGH RISK mainly due to",
        f"elevated '{top[0]}' values relative to the baseline.",
    ]
    return fig, "\n".join(report_lines)


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCE PLOT (feature interaction)
# ─────────────────────────────────────────────────────────────────────────────

def plot_dependence(shap_arr: np.ndarray, X: np.ndarray,
                    feature_names: list, feature: str = 'Speed'):
    if feature not in feature_names:
        return None
    idx  = feature_names.index(feature)
    vals = shap_arr[:, idx]
    feat_vals = X[:, idx]

    fig = go.Figure(go.Scatter(
        x=feat_vals, y=vals,
        mode='markers',
        marker=dict(color=vals, colorscale='plasma', size=4, opacity=0.6,
                    showscale=True, colorbar=dict(title='SHAP'))
    ))
    fig.update_layout(**_L,
                      title=f'📉 SHAP Dependence — {feature}',
                      xaxis_title=feature,
                      yaxis_title='SHAP Value')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_explainable_ai(results: dict):
    """
    Compute SHAP explanations for the best model.
    Returns dict of {name: plotly_fig | text}.
    """
    best_name = results['best']
    model     = results[best_name]['model']
    X_test    = results['X_test']
    feats     = results['features']
    mtype     = 'tree' if 'Forest' in best_name or 'XGBoost' in best_name else 'linear'

    print(f"[SHAP] Computing values for {best_name} …")
    shap_arr, explainer, X_s = compute_shap_values(model, X_test, feats,
                                                    model_type=mtype)

    fi_chart = plot_feature_importance(shap_arr, feats)
    dep_chart= plot_dependence(shap_arr, X_s, feats, 'Speed')

    return {
        'feature_importance': fi_chart,
        'dependence_speed':   dep_chart,
        'shap_array':         shap_arr,
        'X_sample':           X_s,
        'model_type':         mtype,
    }


if __name__ == '__main__':
    from preprocessing   import run_preprocessing
    from adaptive_model  import apply_adaptive_scoring
    from predictive_model import train_and_evaluate

    df  = run_preprocessing()
    df  = apply_adaptive_scoring(df)
    res = train_and_evaluate(df)
    xai = run_explainable_ai(res)
    print("[✔] SHAP explanations generated")
    xai['feature_importance'].write_html('shap_importance.html')
