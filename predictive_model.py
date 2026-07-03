"""
=============================================================================
MODULE 5: PREDICTIVE MODELLING
Project: AI Road Accident Prevention Intelligence System

Trains: Random Forest | Logistic Regression | XGBoost
Auto-selects best model by F1 (macro).
=============================================================================
"""

import numpy as np
import pandas as pd
import joblib, os
from sklearn.model_selection     import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing       import StandardScaler, LabelEncoder
from sklearn.ensemble            import RandomForestClassifier
from sklearn.linear_model        import LogisticRegression
from sklearn.metrics             import (accuracy_score, precision_score, recall_score,
                                         f1_score, confusion_matrix, classification_report,
                                         roc_curve, auc)
from xgboost                     import XGBClassifier
import plotly.express            as px
import plotly.graph_objects      as go
from plotly.subplots             import make_subplots
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = 'best_model.pkl'
SCALER_PATH= 'feature_scaler.pkl'

FEATURE_COLS = [
    'Speed', 'Fatigue', 'TrafficDensity', 'Distraction',
    'Visibility', 'MobileUsage',
    'Weather_enc', 'RoadCondition_enc', 'TimeOfDay_enc',
    'VehicleCondition_enc',
]
TARGET_COL   = 'Severity'

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _encode_target(df):
    le = LabelEncoder()
    y  = le.fit_transform(df[TARGET_COL].astype(str))
    return y, le


def _prepare_xy(df):
    """Select feature + target columns; encode if needed."""
    d = df.copy()

    # Encode categoricals if raw strings present
    for col in ['Weather', 'RoadCondition', 'TimeOfDay', 'VehicleCondition', 'Country']:
        enc_col = col + '_enc'
        if enc_col not in d.columns and col in d.columns:
            le = LabelEncoder()
            d[enc_col] = le.fit_transform(d[col].astype(str))

    feats = [c for c in FEATURE_COLS if c in d.columns]
    X = d[feats].fillna(0).values
    y, le = _encode_target(d)
    return X, y, feats, le


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

def build_models():
    return {
        'Random Forest': RandomForestClassifier(
                          n_estimators=100,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=2,
                          n_jobs=-1,
                           random_state=42

        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced',
            solver='lbfgs', random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=7,
            learning_rate=0.1, use_label_encoder=False,
            eval_metric='mlogloss', random_state=42, n_jobs=-1
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame) -> dict:
    print("\n" + "="*60)
    print("  MODULE 5 — PREDICTIVE MODELLING")
    print("="*60)

    X, y, feats, le = _prepare_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)
    joblib.dump(scaler, SCALER_PATH)

    models  = build_models()
    results = {}

    for name, model in models.items():
        print(f"\n  Training: {name} …")
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        proba = model.predict_proba(X_te) if hasattr(model, 'predict_proba') else None

        acc  = accuracy_score(y_te, preds)
        prec = precision_score(y_te, preds, average='macro', zero_division=0)
        rec  = recall_score(y_te, preds, average='macro', zero_division=0)
        f1   = f1_score(y_te, preds, average='macro', zero_division=0)
        cm   = confusion_matrix(y_te, preds)
        cv   = cross_val_score(model, X, y, cv=StratifiedKFold(3), scoring='f1_macro')

        results[name] = {
            'model':     model,
            'accuracy':  round(acc,  4),
            'precision': round(prec, 4),
            'recall':    round(rec,  4),
            'f1':        round(f1,   4),
            'cv_f1':     round(cv.mean(), 4),
            'cm':        cm,
            'proba':     proba,
            'y_test':    y_te,
            'preds':     preds,
            'classes':   le.classes_,
        }
        print(f"    Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  CV-F1={cv.mean():.3f}")

    # Auto-select best
    best_name = max(results, key=lambda k: results[k]['f1'])
    results['best'] = best_name
    print(f"\n  ✔ Best Model: {best_name}  (F1={results[best_name]['f1']})")

    # Persist
    best_model = results[best_name]['model']
    joblib.dump({'model': best_model, 'scaler': scaler,
                 'features': feats, 'label_enc': le}, MODEL_PATH)
    print(f"  ✔ Saved → {MODEL_PATH}")

    results['scaler']   = scaler
    results['features'] = feats
    results['label_enc']= le
    results['X_test']   = X_te
    results['y_test_raw']= y_te
    print("="*60 + "\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ON NEW DATA
# ─────────────────────────────────────────────────────────────────────────────

def predict_risk(input_dict: dict, model_path=MODEL_PATH):
    """Load saved model and predict severity for a single observation."""
    bundle = joblib.load(model_path)
    model, scaler, feats, le = (
        bundle['model'], bundle['scaler'],
        bundle['features'], bundle['label_enc']
    )
    row = pd.DataFrame([input_dict])
    for col in ['Weather', 'RoadCondition', 'TimeOfDay', 'VehicleCondition', 'Country']:
        enc_col = col + '_enc'
        if enc_col in feats and col in row.columns:
            le_col = LabelEncoder()
            row[enc_col] = le_col.fit_transform(row[col].astype(str))

    X   = row[[c for c in feats if c in row.columns]].fillna(0).values
    X   = scaler.transform(X)
    cls = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
    label = le.inverse_transform([cls])[0]
    return {'prediction': label,
            'probabilities': dict(zip(le.classes_, proba)) if proba is not None else {}}


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

BG = '#0a0e1a'; ACCENT = '#00d4ff'; FONT = '#e0e8ff'; GRID = '#1e2535'
_L = dict(paper_bgcolor=BG, plot_bgcolor=BG,
          font=dict(color=FONT, family='Orbitron, monospace'),
          title_font=dict(size=16, color=ACCENT))


def plot_confusion_matrix(res: dict, model_name: str):
    r   = res[model_name]
    cm  = r['cm']
    cls = r['classes']
    fig = go.Figure(go.Heatmap(
        z=cm, x=cls, y=cls,
        colorscale='Blues',
        text=cm, texttemplate='%{text}',
        showscale=True
    ))
    fig.update_layout(**_L, title=f'Confusion Matrix — {model_name}',
                      xaxis_title='Predicted', yaxis_title='Actual')
    return fig


def plot_model_comparison(res: dict):
    names = [k for k in res if k not in ('best','scaler','features','label_enc','X_test','y_test_raw')]
    metrics = ['accuracy','precision','recall','f1']
    rows = [{'Model': n, **{m: res[n][m] for m in metrics}} for n in names]
    df_m = pd.DataFrame(rows).melt(id_vars='Model', var_name='Metric', value_name='Score')
    fig  = px.bar(df_m, x='Metric', y='Score', color='Model',
                  barmode='group',
                  color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(**_L, title='📊 Model Performance Comparison')
    return fig


def plot_roc_curves(res: dict):
    fig = go.Figure()
    colors = ['#00d4ff','#ff6b6b','#2ecc71']
    for i, name in enumerate([k for k in res if k not in ('best','scaler','features','label_enc','X_test','y_test_raw')]):
        r = res[name]
        if r['proba'] is None:
            continue
        from sklearn.preprocessing import label_binarize
        n_cls = len(r['classes'])
        y_bin = label_binarize(r['y_test'], classes=list(range(n_cls)))
        fpr, tpr, _ = roc_curve(y_bin.ravel(), r['proba'].ravel())
        area = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC={area:.3f})',
                                 line=dict(color=colors[i], width=2)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                             line=dict(dash='dash', color='gray'), name='Random'))
    fig.update_layout(**_L, title='📈 ROC Curves',
                      xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    return fig


if __name__ == '__main__':
    from preprocessing import run_preprocessing
    df = run_preprocessing()
    from adaptive_model import apply_adaptive_scoring
    df = apply_adaptive_scoring(df)
    results = train_and_evaluate(df)
    print("Best model:", results['best'])
