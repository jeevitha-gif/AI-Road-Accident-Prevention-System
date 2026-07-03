🛡️ AI Road Accident Prevention Intelligence System

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

   
## Output Screenshots

### Dashboard
<img width="1366" height="768" alt="Screenshot 2026-06-27 145454" src="https://github.com/user-attachments/assets/1044d160-f66b-4e80-b0aa-4780cd171d1c" />


### Exploratory Data Analysis
<img width="888" height="450" alt="newplot (2)" src="https://github.com/user-attachments/assets/7878088a-bab5-4e2d-968d-2c10b38934e4" />
<img width="888" height="450" alt="newplot (31)" src="https://github.com/user-attachments/assets/ec1409e1-655b-406d-8e95-2c26a774292d" />
<img width="888" height="450" alt="newplot (28)" src="https://github.com/user-attachments/assets/482a6aeb-869e-4f2f-9205-9f7b61bd4fab" />




### Model Performance
<img width="1366" height="768" alt="Screenshot 2026-07-03 121937" src="https://github.com/user-attachments/assets/e6800311-019b-4d5c-8197-d15abaa2565a" />
<img width="436" height="450" alt="newplot (32)" src="https://github.com/user-attachments/assets/34d6ff7d-7d05-48b7-b5ba-269307a936a4" />
<img width="436" height="450" alt="newplot (33)" src="https://github.com/user-attachments/assets/a9e3e499-53ee-483f-a53a-569250efc23f" />
<img width="888" height="450" alt="newplot (35)" src="https://github.com/user-attachments/assets/40cc39ea-237f-47e5-a9bb-f8acf84f31d4" />




### Explainable AI (SHAP)
<img width="888" height="450" alt="newplot (38)" src="https://github.com/user-attachments/assets/f890b3f4-337a-4085-b657-eff2e6017b8d" />
<img width="916" height="497" alt="Screenshot 2026-07-03 122346" src="https://github.com/user-attachments/assets/a5dc6bb3-8b43-460b-b86c-9a637afad481" />

### Recommendation
<img width="835" height="602" alt="image" src="https://github.com/user-attachments/assets/dbb03a2a-0504-4978-9390-92bb78b313ee" />

### Live Monitor
<img width="939" height="597" alt="Screenshot 2026-07-03 122650" src="https://github.com/user-attachments/assets/51b26ca2-3632-4e3e-b29a-03524a55b296" />


---

## 🎓 Project Info

- **Type:** MCA Final Year Project 
- **Stack:** Python · Streamlit · Scikit-learn · XGBoost · SHAP · Plotly
- **Dataset:** 100,000+ synthetic road telemetry records
