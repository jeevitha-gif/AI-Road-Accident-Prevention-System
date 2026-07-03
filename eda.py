"""
=============================================================================
MODULE 2: EXPLORATORY DATA ANALYSIS
Project: AI Road Accident Prevention Intelligence System
=============================================================================
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Colour palette ──────────────────────────────────────────────────────────
PALETTE   = px.colors.qualitative.Bold
ACCENT    = '#00d4ff'
BG        = '#0a0e1a'
GRID      = '#1e2535'
FONT_COL  = '#e0e8ff'

LAYOUT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=FONT_COL, family='Orbitron, monospace'),
    title_font=dict(size=18, color=ACCENT),
    legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0),
)


def _apply_dark(fig, title=''):
    fig.update_layout(**LAYOUT_BASE, title=title)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. MONTHLY ACCIDENT TREND
# ─────────────────────────────────────────────────────────────────────────────

def plot_monthly_trend(df: pd.DataFrame):
    d = df.copy()
    d['Month'] = pd.to_datetime(d['Date'], errors='coerce').dt.to_period('M').astype(str)
    monthly = d.groupby('Month').agg(
        AvgRisk=('AccidentRisk', 'mean'),
        Count  =('AccidentRisk', 'count')
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=monthly['Month'], y=monthly['AvgRisk'],
        mode='lines+markers', name='Avg Risk',
        line=dict(color=ACCENT, width=2),
        marker=dict(size=5)
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=monthly['Month'], y=monthly['Count'],
        name='Incidents', opacity=0.4,
        marker_color='#ff6b6b'
    ), secondary_y=True)
    fig.update_yaxes(title_text='Avg Risk Score', secondary_y=False)
    fig.update_yaxes(title_text='Incident Count',  secondary_y=True)
    return _apply_dark(fig, '📈 Monthly Accident Trend')


# ─────────────────────────────────────────────────────────────────────────────
# 2. WEATHER IMPACT
# ─────────────────────────────────────────────────────────────────────────────

def plot_weather_impact(df: pd.DataFrame):
    grp = df.groupby('Weather')['AccidentRisk'].mean().reset_index().sort_values('AccidentRisk', ascending=False)
    fig = px.bar(grp, x='Weather', y='AccidentRisk',
                 color='AccidentRisk',
                 color_continuous_scale='plasma',
                 labels={'AccidentRisk': 'Avg Risk Score'})
    return _apply_dark(fig, '🌦 Weather Impact on Accident Risk')


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAFFIC DENSITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def plot_traffic_density(df: pd.DataFrame):
    fig = px.violin(
        df.sample(min(20_000, len(df))),
        x='TrafficDensity', y='AccidentRisk',
        color='TimeOfDay',
        box=True, points=False,
        color_discrete_sequence=PALETTE,
        labels={'AccidentRisk': 'Risk Score', 'TrafficDensity': 'Traffic Density'}
    )
    return _apply_dark(fig, '🚦 Traffic Density vs Risk by Time of Day')


# ─────────────────────────────────────────────────────────────────────────────
# 4. SEVERITY DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_severity_distribution(df: pd.DataFrame):
    counts = df['Severity'].value_counts().reset_index()
    counts.columns = ['Severity', 'Count']
    color_map = {'Low': '#2ecc71', 'Medium': '#f1c40f', 'High': '#e67e22', 'Critical': '#e74c3c'}
    fig = px.pie(counts, names='Severity', values='Count',
                 color='Severity', color_discrete_map=color_map,
                 hole=0.45)
    fig.update_traces(textfont_size=13, pull=[0, 0, 0.05, 0.1])
    return _apply_dark(fig, '🎯 Accident Severity Distribution')


# ─────────────────────────────────────────────────────────────────────────────
# 5. CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame):
    num_cols = ['Speed', 'Fatigue', 'TrafficDensity', 'Distraction',
                'Visibility', 'MobileUsage', 'AccidentRisk']
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()
    fig  = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale='RdBu_r', zmid=0,
        text=corr.values.round(2), texttemplate='%{text}',
        showscale=True
    ))
    return _apply_dark(fig, '🔥 Feature Correlation Heatmap')


# ─────────────────────────────────────────────────────────────────────────────
# 6. DRIVER BEHAVIOUR RISK PATTERN
# ─────────────────────────────────────────────────────────────────────────────

def plot_driver_behavior(df: pd.DataFrame):
    sample = df.sample(min(10_000, len(df)))
    fig = px.scatter(
        sample, x='Fatigue', y='Speed',
        color='AccidentRisk', size='Distraction',
        color_continuous_scale='inferno',
        opacity=0.6,
        labels={'AccidentRisk': 'Risk Score'}
    )
    return _apply_dark(fig, '🧠 Driver Behaviour Risk Pattern')


# ─────────────────────────────────────────────────────────────────────────────
# 7. RISK DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_risk_distribution(df: pd.DataFrame):
    fig = px.histogram(
        df, x='AccidentRisk', nbins=60,
        color_discrete_sequence=[ACCENT],
        marginal='box',
        labels={'AccidentRisk': 'Accident Risk Score'}
    )
    return _apply_dark(fig, '📊 Accident Risk Score Distribution')


# ─────────────────────────────────────────────────────────────────────────────
# 8. HIGH-RISK ZONES (Country × Weather)
# ─────────────────────────────────────────────────────────────────────────────

def plot_high_risk_zones(df: pd.DataFrame):
    grp = df.groupby(['Country', 'Weather'])['AccidentRisk'].mean().reset_index()
    pivot = grp.pivot(index='Country', columns='Weather', values='AccidentRisk').fillna(0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='YlOrRd',
        text=pivot.values.round(2),
        texttemplate='%{text}',
    ))
    return _apply_dark(fig, '🗺 High-Risk Zones — Country × Weather')


# ─────────────────────────────────────────────────────────────────────────────
# MASTER EDA FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> dict:
    """Return a dict of {name: plotly_figure}."""
    return {
        'monthly_trend':     plot_monthly_trend(df),
        'weather_impact':    plot_weather_impact(df),
        'traffic_density':   plot_traffic_density(df),
        'severity_dist':     plot_severity_distribution(df),
        'correlation':       plot_heatmap(df),
        'driver_behavior':   plot_driver_behavior(df),
        'risk_distribution': plot_risk_distribution(df),
        'high_risk_zones':   plot_high_risk_zones(df),
    }


if __name__ == '__main__':
    from preprocessing import run_preprocessing
    df   = run_preprocessing()
    figs = run_eda(df)
    print(f"[✔] EDA complete — {len(figs)} charts generated")
    # Save one as HTML sample
    figs['monthly_trend'].write_html('eda_monthly_trend.html')
    print("[✔] Sample chart saved → eda_monthly_trend.html")
