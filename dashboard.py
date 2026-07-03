"""
=============================================================================
MODULE 8: PREMIUM PATENT-STYLE STREAMLIT DASHBOARD
Project: AI Road Accident Prevention Intelligence System
Theme: Dark Cyber-Blue Neon AI
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time, json, os, warnings
warnings.filterwarnings('ignore')

# ── Local modules ─────────────────────────────────────────────────────────────
from preprocessing    import run_preprocessing, load_dataset
from adaptive_model   import (compute_adaptive_risk_score, apply_adaptive_scoring,
                              WEATHER_RISK, ROAD_RISK, TIME_RISK)
from eda              import run_eda
from predictive_model import train_and_evaluate, plot_confusion_matrix, plot_model_comparison, plot_roc_curves
from explainable_ai   import run_explainable_ai, explain_single_prediction, compute_shap_values, plot_feature_importance
from recommendation   import generate_recommendations, print_recommendations

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Road Accident Prevention Intelligence System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM CSS — Cyber-Blue Neon Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

/* ── Root variables ── */
:root {
  --bg:      #050a14;
  --panel:   #0b1325;
  --border:  #0e2a4a;
  --accent:  #00d4ff;
  --accent2: #0080ff;
  --danger:  #ff3860;
  --warn:    #ffdd57;
  --safe:    #23d160;
  --text:    #c8d8f0;
  --dim:     #607090;
}

/* ── Global ── */
html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Rajdhani', sans-serif !important;
}

/* ── Header bar ── */
.main-header {
  background: linear-gradient(135deg, #050a14 0%, #0a1628 50%, #050a14 100%);
  border-bottom: 2px solid var(--accent);
  padding: 1.5rem 2rem;
  margin-bottom: 1.5rem;
  border-radius: 0 0 12px 12px;
  position: relative;
  overflow: hidden;
}
.main-header::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(ellipse, rgba(0,212,255,0.04) 0%, transparent 70%);
  pointer-events: none;
}
.main-title {
  font-family: 'Orbitron', monospace !important;
  font-size: 2rem !important;
  font-weight: 900 !important;
  color: var(--accent) !important;
  text-shadow: 0 0 20px rgba(0,212,255,0.5);
  margin: 0;
  letter-spacing: 2px;
}
.main-subtitle {
  font-size: 0.85rem;
  color: var(--dim);
  font-style: italic;
  letter-spacing: 1px;
}

/* ── KPI Cards ── */
.kpi-card {
  background: linear-gradient(135deg, var(--panel) 0%, #0d1c35 100%);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem 1.5rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 32px rgba(0,212,255,0.2);
}
.kpi-card::after {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 3px;
  background: linear-gradient(90deg, var(--accent2), var(--accent));
}
.kpi-value {
  font-family: 'Orbitron', monospace;
  font-size: 2rem;
  font-weight: 700;
  color: var(--accent);
  text-shadow: 0 0 12px rgba(0,212,255,0.4);
}
.kpi-label {
  font-size: 0.75rem;
  color: var(--dim);
  text-transform: uppercase;
  letter-spacing: 2px;
  margin-top: 0.3rem;
}
.kpi-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }

/* ── Risk badge ── */
.risk-safe     { color: var(--safe);   font-weight:700; }
.risk-medium   { color: var(--warn);   font-weight:700; }
.risk-high     { color: #ff8c42;       font-weight:700; }
.risk-critical { color: var(--danger); font-weight:700;
                 text-shadow: 0 0 8px rgba(255,56,96,0.6); animation: pulse 1s infinite; }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.6; } }

/* ── Recommendation cards ── */
.rec-card {
  border-radius: 10px;
  padding: 1rem 1.5rem;
  margin: 0.5rem 0;
  border-left: 4px solid;
}
.rec-critical { background: rgba(255,56,96,0.08);  border-color: var(--danger); }
.rec-alert    { background: rgba(255,221,87,0.08); border-color: var(--warn);  }
.rec-warning  { background: rgba(0,212,255,0.08);  border-color: var(--accent);}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #060c1a 0%, #08111f 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
  color: var(--accent) !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 0.75rem !important;
  letter-spacing: 1px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: var(--panel) !important;
  border-radius: 8px;
  padding: 4px;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Orbitron', monospace !important;
  font-size: 0.65rem !important;
  color: var(--dim) !important;
  border-radius: 6px !important;
  padding: 0.5rem 0.8rem !important;
  letter-spacing: 1px;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
  color: #fff !important;
}

/* ── Metrics override ── */
[data-testid="metric-container"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 0.8rem !important;
}

/* ── Misc ── */
.section-title {
  font-family: 'Orbitron', monospace;
  font-size: 1rem;
  color: var(--accent);
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.4rem;
  margin: 1.5rem 0 1rem;
  letter-spacing: 2px;
}
.info-box {
  background: rgba(0,212,255,0.06);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA CACHING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_data():
    df = run_preprocessing()
    df = apply_adaptive_scoring(df)
    return df

@st.cache_resource(show_spinner=False)
def get_model_results(df):
    return train_and_evaluate(df)

@st.cache_resource(show_spinner=False)
def get_xai(_results):
    return run_explainable_ai(_results)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <div class="main-title">🛡️ AI ROAD ACCIDENT PREVENTION INTELLIGENCE SYSTEM</div>
  <div class="main-subtitle">
    A Self-Adaptive Context-Aware Predictive Intelligence Framework for Real-Time Road Accident Prevention
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("⚡ Initialising AI engines…"):
    df = get_data()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; color:#00d4ff; font-size:1rem;
                border-bottom:1px solid #0e2a4a; padding-bottom:0.5rem; margin-bottom:1rem;">
      ⚙️ CONTROL PANEL
    </div>""", unsafe_allow_html=True)

    f_country  = st.multiselect("🌍 COUNTRY",
                    sorted(df['Country'].dropna().unique()),
                    default=sorted(df['Country'].dropna().unique()))
    f_weather  = st.multiselect("🌦 WEATHER",
                    sorted(df['Weather'].dropna().unique()),
                    default=sorted(df['Weather'].dropna().unique()))
    f_time     = st.multiselect("⏰ TIME OF DAY",
                    sorted(df['TimeOfDay'].dropna().unique()),
                    default=sorted(df['TimeOfDay'].dropna().unique()))
    f_risk     = st.multiselect("⚠️ RISK LEVEL",
                    ['Safe','Medium Risk','High Risk','Critical Risk'],
                    default=['Safe','Medium Risk','High Risk','Critical Risk'])
    f_traffic  = st.slider("🚦 MAX TRAFFIC DENSITY", 1, 10, 10)

    st.markdown("---")
   

# Apply filters
filt = (
    df['Country'].isin(f_country) &
    df['Weather'].isin(f_weather) &
    df['TimeOfDay'].isin(f_time) &
    df['RiskLabel'].isin(f_risk) &
    (df['TrafficDensity'] <= f_traffic)
)
dff = df[filt].copy()

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────

total       = len(dff)
avg_risk    = dff['AccidentRisk'].mean()
critical_n  = (dff['RiskLabel'] == 'Critical Risk').sum()
safety_idx  = round((1 - avg_risk / 10) * 100, 1)

kpi_cols = st.columns(5)
kpis = [
    ("📊", f"{total:,}", "TOTAL RECORDS"),
    ("⚠️", f"{avg_risk:.2f}", "AVG RISK SCORE"),
    ("🚨", f"{critical_n:,}", "CRITICAL CASES"),
    ("🎯", "Pending", "PRED. ACCURACY"),
    ("🛡️", f"{safety_idx}%", "SAFETY INDEX"),
]
for col, (icon, val, label) in zip(kpi_cols, kpis):
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-value">{val}</div>
      <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📋 Data Quality",
    "📊 EDA Analysis",
    "⚙️ Adaptive Engine",
    "🤖 Prediction",
    "🧠 Explainable AI",
    "💡 Recommendations",
    "🔴 Live Monitor",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">📋 DATA QUALITY REPORT</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",    f"{len(df):,}")
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Null Values",   df.isnull().sum().sum())
    c4.metric("Duplicates",    df.duplicated().sum())

    st.markdown('<div class="section-title">COLUMN STATISTICS</div>', unsafe_allow_html=True)
    num_desc = dff.describe().T.round(3)
    st.dataframe(num_desc, use_container_width=True)

    st.markdown('<div class="section-title">MISSING VALUE MATRIX</div>', unsafe_allow_html=True)
    null_data = df.isnull().sum().reset_index()
    null_data.columns = ['Feature', 'Null Count']
    null_data['Null %'] = (null_data['Null Count'] / len(df) * 100).round(2)
    null_data = null_data[null_data['Null Count'] > 0]
    if len(null_data):
        fig = px.bar(null_data, x='Feature', y='Null %',
                     color='Null %', color_continuous_scale='reds',
                     title='Missing Value Percentage by Feature')
        fig.update_layout(paper_bgcolor='#0a0e1a', plot_bgcolor='#0a0e1a',
                          font_color='#e0e8ff')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅  No missing values in filtered dataset.")

    st.markdown('<div class="section-title">RISK LABEL DISTRIBUTION</div>', unsafe_allow_html=True)
    rl = dff['RiskLabel'].value_counts().reset_index()
    rl.columns = ['RiskLabel', 'Count']
    fig2 = px.pie(rl, names='RiskLabel', values='Count',
                  color='RiskLabel',
                  color_discrete_map={
                      'Safe': '#23d160', 'Medium Risk': '#ffdd57',
                      'High Risk': '#ff8c42', 'Critical Risk': '#ff3860'
                  }, hole=0.4)
    fig2.update_layout(paper_bgcolor='#0a0e1a', font_color='#e0e8ff')
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">📊 EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)

    with st.spinner("Generating EDA charts…"):
        eda_figs = run_eda(dff.head(30_000))  # cap for speed

    for title, fig in eda_figs.items():
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ADAPTIVE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">⚙️ ADAPTIVE RISK SCORING ENGINE</div>', unsafe_allow_html=True)

    

    # Interactive tester
    st.markdown('<div class="section-title">LIVE RISK CALCULATOR</div>', unsafe_allow_html=True)
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        t_speed   = st.slider("Speed (km/h)",   0, 180, 80)
        t_fatigue = st.slider("Fatigue",         0, 10,  3)
        t_traffic = st.slider("Traffic Density", 1, 10,  5)
    with rc2:
        t_dist    = st.slider("Distraction",     0, 10,  2)
        t_vis     = st.slider("Visibility",      0, 10,  7)
        t_mobile  = st.selectbox("Mobile Usage", [0, 1])
    with rc3:
        t_weather = st.selectbox("Weather",    ['Clear','Rainy','Foggy','Stormy','Snowy'])
        t_road    = st.selectbox("Road Cond.", ['Good','Wet','Damaged','Icy','Under Construction'])
        t_time    = st.selectbox("Time of Day",['Morning','Afternoon','Evening','Night'])
        t_vehicle = st.selectbox("Vehicle",    ['Good','Fair','Poor'])

    if st.button("⚡ COMPUTE ADAPTIVE RISK SCORE", type="primary"):
        inp = dict(Speed=t_speed, Fatigue=t_fatigue, Weather=t_weather,
                   TrafficDensity=t_traffic, RoadCondition=t_road,
                   Distraction=t_dist, Visibility=t_vis, TimeOfDay=t_time,
                   MobileUsage=t_mobile, VehicleCondition=t_vehicle)
        result = compute_adaptive_risk_score(inp)
        score  = result['AdaptiveRiskScore']
        label  = result['RiskLabel']

        # Gauge
        color_map = {'Safe':'#23d160','Medium Risk':'#ffdd57',
                     'High Risk':'#ff8c42','Critical Risk':'#ff3860'}
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={'reference': 5},
            gauge={
                'axis':   {'range':[0,10], 'tickcolor':'#00d4ff'},
                'bar':    {'color': color_map.get(label,'#00d4ff')},
                'steps':  [
                    {'range':[0,3],   'color':'rgba(35,209,96,0.2)'},
                    {'range':[3,5],   'color':'rgba(255,221,87,0.2)'},
                    {'range':[5,7],   'color':'rgba(255,140,66,0.2)'},
                    {'range':[7,10],  'color':'rgba(255,56,96,0.2)'},
                ],
                'threshold': {'line':{'color':'white','width':4},'value':score}
            },
            title={'text': f"Adaptive Risk Score — {label}",
                   'font':{'color':'#00d4ff','family':'Orbitron','size':16}}
        ))
        gauge_fig.update_layout(paper_bgcolor='#0a0e1a', font_color='#e0e8ff',
                                height=350)
        st.plotly_chart(gauge_fig, use_container_width=True)

        # Weights breakdown
        wt = result['Weights']
        wf = result['FeatureScores']
        wc1, wc2 = st.columns(2)
        with wc1:
            wfig = px.bar(x=list(wt.keys()), y=list(wt.values()),
                          color=list(wt.values()), color_continuous_scale='plasma',
                          title="Dynamic Weights")
            wfig.update_layout(paper_bgcolor='#0a0e1a', font_color='#e0e8ff',
                               plot_bgcolor='#0a0e1a')
            st.plotly_chart(wfig, use_container_width=True)
        with wc2:
            sfig = px.bar(x=list(wf.keys()), y=list(wf.values()),
                          color=list(wf.values()), color_continuous_scale='reds',
                          title="Normalised Factor Scores")
            sfig.update_layout(paper_bgcolor='#0a0e1a', font_color='#e0e8ff',
                               plot_bgcolor='#0a0e1a')
            st.plotly_chart(sfig, use_container_width=True)

    # Adaptive score distribution
    st.markdown('<div class="section-title">ADAPTIVE RISK SCORE DISTRIBUTION</div>', unsafe_allow_html=True)
    fig_ars = px.histogram(dff, x='AdaptiveRiskScore', nbins=50,
                           color='RiskLabel',
                           color_discrete_map={
                               'Safe':'#23d160','Medium Risk':'#ffdd57',
                               'High Risk':'#ff8c42','Critical Risk':'#ff3860'},
                           barmode='overlay', opacity=0.75)
    fig_ars.update_layout(paper_bgcolor='#0a0e1a', plot_bgcolor='#0a0e1a',
                          font_color='#e0e8ff')
    st.plotly_chart(fig_ars, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTION ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">🤖 PREDICTIVE MODELLING</div>', unsafe_allow_html=True)

    if st.button("🚀 TRAIN ALL MODELS", type="primary"):
        with st.spinner("Training Random Forest | XGBoost | Logistic Regression…"):
            ml_results = get_model_results(df)
            st.session_state['ml_results'] = ml_results

    if 'ml_results' in st.session_state:
        ml_results = st.session_state['ml_results']
        best       = ml_results['best']

        # Update accuracy KPI
        st.success(f"✅  Best Model: **{best}**  |  F1 = {ml_results[best]['f1']}")

        # Metrics table
        model_names = [k for k in ml_results if k not in
                       ('best','scaler','features','label_enc','X_test','y_test_raw')]
        rows = []
        for n in model_names:
            r = ml_results[n]
            rows.append({'Model':n,'Accuracy':r['accuracy'],
                         'Precision':r['precision'],
                         'Recall':r['recall'],'F1':r['f1'],'CV F1':r['cv_f1']})
        st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

        mc1, mc2 = st.columns(2)
        with mc1:
            st.plotly_chart(plot_model_comparison(ml_results), use_container_width=True)
        with mc2:
            st.plotly_chart(plot_roc_curves(ml_results), use_container_width=True)

        # Confusion matrix selector
        sel = st.selectbox("Confusion Matrix for:", model_names)
        st.plotly_chart(plot_confusion_matrix(ml_results, sel), use_container_width=True)
    else:
        st.info("Click 'TRAIN ALL MODELS' to start training.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPLAINABLE AI
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">🧠 EXPLAINABLE AI — SHAP ANALYSIS</div>', unsafe_allow_html=True)

    if 'ml_results' not in st.session_state:
        st.warning("Please train models first (Prediction tab).")
    else:
        ml_results = st.session_state['ml_results']
        with st.spinner("Computing SHAP values…"):
            xai = get_xai(ml_results)

        st.plotly_chart(xai['feature_importance'], use_container_width=True)
        if xai['dependence_speed']:
            st.plotly_chart(xai['dependence_speed'], use_container_width=True)

        st.markdown('<div class="section-title">SINGLE PREDICTION EXPLANATION</div>', unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            s_speed   = st.slider("Speed",   0, 180, 120, key='xai_spd')
            s_fatigue = st.slider("Fatigue", 0, 10, 7, key='xai_fat')
            s_traffic = st.slider("Traffic", 1, 10, 8, key='xai_trf')
        with sc2:
            s_dist  = st.slider("Distraction", 0, 10, 6, key='xai_dst')
            s_vis   = st.slider("Visibility",  0, 10, 3, key='xai_vis')
            s_mob   = st.selectbox("Mobile", [0,1], key='xai_mob')
        with sc3:
            s_wea = st.selectbox("Weather", ['Clear','Rainy','Foggy','Stormy','Snowy'], key='xai_w')
            s_road = st.selectbox( "Road", ['Good','Wet','Damaged','Icy','Under Construction'],key='xai_r')
            s_time = st.selectbox( "TimeOfDay",  ['Morning','Afternoon','Evening','Night'],  key='xai_t'  )
            s_vehicle = st.selectbox("Vehicle Condition", ['Good','Fair','Poor'], key='xai_vehicle' )
        
       
       
   

        if st.button("🔍 EXPLAIN PREDICTION"):
            best_name  = ml_results['best']
            best_model = ml_results[best_name]['model']
            scaler     = ml_results['scaler']
            feats      = ml_results['features']
            mtype      = 'tree' if 'Forest' in best_name or 'XGBoost' in best_name else 'linear'
            inp2 = dict(Speed=s_speed, Fatigue=s_fatigue, Weather=s_wea,
                        TrafficDensity=s_traffic, RoadCondition=s_road,
                        Distraction=s_dist, Visibility=s_vis, TimeOfDay=s_time,
                        MobileUsage=s_mob)
            fig_xai, report = explain_single_prediction(
                best_model, scaler, feats, inp2, model_type=mtype
            )
            st.plotly_chart(fig_xai, use_container_width=True)
            st.code(report, language='text')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">💡 INTELLIGENT PREVENTION RECOMMENDATIONS</div>',
                unsafe_allow_html=True)

    rr1, rr2, rr3 = st.columns(3)
    with rr1:
        r_speed   = st.slider("Speed",   0, 180, 130, key='rec_spd')
        r_fatigue = st.slider("Fatigue", 0, 10, 8, key='rec_fat')
        r_traffic = st.slider("Traffic", 1, 10, 9, key='rec_trf')
    with rr2:
        r_dist    = st.slider("Distraction", 0, 10, 7, key='rec_dst')
        r_vis     = st.slider("Visibility",  0, 10, 2, key='rec_vis')
        r_mob     = st.selectbox("Mobile", [1, 0], key='rec_mob')
    with rr3:
        r_wea     = st.selectbox("Weather",   ['Stormy','Foggy','Rainy','Clear','Snowy'], key='rec_w')
        r_road    = st.selectbox("Road",      ['Icy','Damaged','Wet','Good','Under Construction'], key='rec_r')
        r_time    = st.selectbox("TimeOfDay", ['Night','Evening','Morning','Afternoon'], key='rec_t')
        r_veh     = st.selectbox("Vehicle",   ['Poor','Fair','Good'], key='rec_v')

    if st.button("💡 GENERATE RECOMMENDATIONS", type="primary"):
        r_inp = dict(Speed=s_speed,
                      
    Fatigue=s_fatigue,
    Weather=s_wea,
    TrafficDensity=s_traffic,
    RoadCondition=s_road,
    Distraction=s_dist,
    Visibility=s_vis,
    TimeOfDay=s_time,
    MobileUsage=s_mob,
    VehicleCondition=s_vehicle
)
        ar = compute_adaptive_risk_score(r_inp)
        recs = generate_recommendations(r_inp, ar['AdaptiveRiskScore'])

        sev_cls = {'Critical':'rec-critical','Alert':'rec-alert','Warning':'rec-warning'}
        for rec in recs:
            css = sev_cls.get(rec.severity, 'rec-warning')
            st.markdown(f"""
            <div class="rec-card {css}">
              <b>{rec.icon} [{rec.severity}] {rec.headline}</b><br>
              <span style="color:#9ab">{rec.detail}</span><br>
              <span style="color:#00d4ff">➤ {rec.action}</span>
            </div>""", unsafe_allow_html=True)

        if not recs:
            st.success("✅ No significant risks detected. Drive safely!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — LIVE RISK MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-title">🔴 LIVE RISK MONITOR</div>', unsafe_allow_html=True)
    st.info("Simulating real-time telemetry stream (synthetic data)")

    placeholder  = st.empty()
    chart_holder = st.empty()

    history = []
    if st.button("▶ START LIVE MONITOR (20 ticks)"):
        for tick in range(20):
            row = df.sample(1).iloc[0].to_dict()
            res = compute_adaptive_risk_score(row)
            score = res['AdaptiveRiskScore']
            label = res['RiskLabel']
            history.append({'Tick': tick+1, 'Score': score, 'Label': label})

            cls_map = {'Safe':'🟢','Medium Risk':'🟡','High Risk':'🟠','Critical Risk':'🔴'}
            placeholder.markdown(f"""
            <div class="kpi-card" style="margin:1rem 0;">
              <div class="kpi-icon">{cls_map.get(label,'⚪')}</div>
              <div class="kpi-value">{score:.2f}</div>
              <div class="kpi-label">{label} — Tick {tick+1}</div>
            </div>""", unsafe_allow_html=True)

            hist_df  = pd.DataFrame(history)
            line_fig = px.line(hist_df, x='Tick', y='Score',
                               color_discrete_sequence=['#00d4ff'],
                               markers=True, title='Live Adaptive Risk Score')
            line_fig.add_hline(y=7, line_dash='dash', line_color='#ff3860',
                               annotation_text='Critical Threshold')
            line_fig.update_layout(paper_bgcolor='#0a0e1a', plot_bgcolor='#0a0e1a',
                                   font_color='#e0e8ff',
                                   xaxis=dict(gridcolor='#1e2535'),
                                   yaxis=dict(gridcolor='#1e2535'))
            chart_holder.plotly_chart(line_fig, use_container_width=True)
            time.sleep(0.4)

        st.success("✅ Live monitoring session complete.")

# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#0e2a4a;">
<div style="text-align:center; color:#607090; font-family:'Rajdhani',sans-serif; font-size:0.75rem;">
  AI Road Accident Prevention Intelligence System &nbsp;|&nbsp;
  MCA Final Year Patent Project &nbsp;|&nbsp;
  <span style="color:#00d4ff;">Powered by Adaptive AI</span>
</div>
""", unsafe_allow_html=True)
