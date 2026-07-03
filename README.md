# 🛡️ AI Road Accident Prevention Intelligence System

**Formal Title:** A Self-Adaptive Context-Aware Predictive Intelligence Framework  
for Real-Time Road Accident Prevention

---

## 📁 Project Structure

```
ai_road_accident/
├── preprocessing.py      # Module 1 — Data pipeline (100K+ synthetic dataset)
├── eda.py                # Module 2 — 8 premium Plotly visualisations
├── adaptive_model.py     # Module 3 — Adaptive Risk Engine
├── predictive_model.py   # Module 5 — RF | XGBoost | LR + auto best-select
├── explainable_ai.py     # Module 6 — SHAP Explainability Engine
├── recommendation.py     # Module 7 — Context-aware Prevention Engine
├── dashboard.py          # Module 8 — Streamlit Dashboard (main entry point)
├── documentation         # Module 9 - PPT presentation
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the dashboard
```bash
streamlit run dashboard.py
```

The first run auto-generates a 100,000-row synthetic dataset. This takes ~30 seconds.

---

## 🧩 Module Overview

| Module | File | Description |
|--------|------|-------------|
| 1 | preprocessing.py | Load/generate data, clean, engineer features |
| 2 | eda.py | 8 premium Plotly charts |
| 3 | adaptive_model.py | self-adaptive risk scoring |
| 5 | predictive_model.py | 3 ML models, auto best-select |
| 6 | explainable_ai.py | SHAP explainability |
| 7 | recommendation.py | Rule-based prevention engine |
| 8 | dashboard.py | 8-tab Streamlit UI |
| 10 | documentation | PPT Presentation |

---

## ⚙️Adaptive Risk Formula

```
AccidentRisk = Ws·Speed + Wf·Fatigue + Ww·Weather
             + Wt·Traffic + Wr·RoadCond + Wd·Distraction + Wv·Visibility
```

Weights **self-adjust** at runtime based on active hazard context:

| Context | Weight Boosted |
|---------|----------------|
| Stormy / Snowy / Foggy | Ww ↑ +0.10 |
| Night driving | Wv ↑ +0.10 |
| Traffic ≥ 8 | Wt ↑ +0.08 |
| Fatigue ≥ 7 | Wf ↑ +0.08 |
| Speed ≥ 120 | Ws ↑ +0.08 |
| Mobile usage | Wd ↑ +0.08 |
| Icy / Damaged road | Wr ↑ +0.06 |

---

## 📊 Dashboard Tabs

1. **Data Quality** — shape, nulls, distributions
2. **EDA Analysis** — 8 interactive charts
3. **Adaptive Engine** — live risk calculator + gauge
4. **Prediction Analytics** — model training + metrics
5. **Explainable AI** — SHAP charts + per-prediction report
6. **Recommendations** — contextual prevention actions
7. **Live Monitor** — simulated real-time telemetry stream


---

## 🎓 Project Info

- **Type:** MCA Final Year Project 
- **Stack:** Python · Streamlit · Scikit-learn · XGBoost · SHAP · Plotly
- **Dataset:** 100,000+ synthetic road telemetry records
