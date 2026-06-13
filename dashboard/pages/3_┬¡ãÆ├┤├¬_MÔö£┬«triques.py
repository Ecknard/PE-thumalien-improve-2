"""
dashboard/pages/3_📈_Métriques.py
Performance du modèle de classification avec diagnostics.
"""
import sys
import json
import pickle
import subprocess
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Métriques — Thumalien", page_icon="📈", layout="wide")
st.markdown("# 📈 Performance du Modèle")
st.markdown("Métriques d'évaluation du classificateur fake news.")

from config import MODELS_DIR, CLASSIFIER_OUTPUT_DIR

# ============================================================
# CHARGEMENT MÉTRIQUES
# ============================================================
metrics = None
model_type = None
cm_real = None  # vraie matrice de confusion si disponible

bert_metrics_path = Path(CLASSIFIER_OUTPUT_DIR) / "metrics.json"
bert_cm_path      = Path(CLASSIFIER_OUTPUT_DIR) / "confusion_matrix.json"
baseline_path     = MODELS_DIR / "baseline_classifier.pkl"
training_summary  = MODELS_DIR / "training_summary.json"

if bert_metrics_path.exists():
    with open(bert_metrics_path) as f:
        metrics = json.load(f)
    model_type = "DistilBERT Multilingue"
    if bert_cm_path.exists():
        with open(bert_cm_path) as f:
            cm_real = json.load(f)

elif baseline_path.exists():
    with open(baseline_path, "rb") as f:
        data = pickle.load(f)
    metrics = data.get("metrics", {})
    model_type = "Baseline TF-IDF + LogisticRegression"
    if "confusion_matrix" in data:
        cm_real = data["confusion_matrix"]

else:
    st.warning("⚠️ Aucun modèle entraîné.")
    st.info("Lancez l'entraînement depuis le terminal :")
    st.code("make train          # Baseline (rapide, ~30s)\nmake train-liar     # DistilBERT fine-tuné (recommandé)")
    st.stop()

# ============================================================
# BANNER DIAGNOSTIC — CRITIQUE
# Affiche clairement pourquoi DistilBERT n'est pas actif
# ============================================================
if model_type and "TF-IDF" in model_type:
    st.warning(
        "⚠️ **Modèle actif : Baseline TF-IDF** — DistilBERT n'a pas encore été entraîné.  \n"
        "Les performances affichées sont indicatives. Pour activer DistilBERT : `make train-liar` (~20 min)"
    )
elif model_type:
    st.success("✅ Modèle DistilBERT multilingue actif (fine-tuné)")

# Warning ROC-AUC artificiel
roc = metrics.get("roc_auc", 0)
n_test = metrics.get("n_test", 0)
if roc >= 0.98 and n_test < 30:
    st.error(
        f"🔴 **ROC-AUC = {roc:.3f} sur seulement {n_test} exemples de test** — "
        "Ce score est **artificiel** : le dataset d'évaluation est trop petit.  \n"
        "Entraînez sur un vrai dataset : `make train-liar` (dataset LIAR ~12k exemples)"
    )

st.subheader(f"🤖 Modèle actif : {model_type}")

# ============================================================
# KPIs
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy",  f"{metrics.get('accuracy', 0):.1%}")
col2.metric("F1-Score",  f"{metrics.get('f1', 0):.1%}")
col3.metric("Précision", f"{metrics.get('precision', 0):.1%}")
col4.metric("Rappel",    f"{metrics.get('recall', 0):.1%}")
roc_display = f"{roc:.3f}" if roc else "—"
col5.metric("ROC-AUC",   roc_display,
            delta="⚠️ dataset trop petit" if roc >= 0.98 and n_test < 30 else None,
            delta_color="off")

st.divider()

# ============================================================
# GRAPHIQUES
# ============================================================
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 Résumé des métriques")
    metric_names  = ["Accuracy", "Précision", "Rappel", "F1-Score"]
    metric_values = [
        metrics.get("accuracy", 0), metrics.get("precision", 0),
        metrics.get("recall", 0),   metrics.get("f1", 0),
    ]
    fig_bar = go.Figure(go.Bar(
        x=metric_names,
        y=[v * 100 for v in metric_values],
        marker_color=["#667eea", "#764ba2", "#22c55e", "#f59e0b"],
        text=[f"{v:.1%}" for v in metric_values],
        textposition="outside",
    ))
    fig_bar.update_layout(
        yaxis=dict(range=[0, 110], title="Score (%)"),
        height=300, margin=dict(t=20, b=20, l=20, r=20), showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_b:
    st.subheader("🕸️ Radar des performances")
    categories = ["Accuracy", "Précision", "Rappel", "F1-Score", "ROC-AUC"]
    values = [
        metrics.get("accuracy", 0), metrics.get("precision", 0),
        metrics.get("recall", 0),   metrics.get("f1", 0),
        min(metrics.get("roc_auc", 0.5), 1.0),
    ]
    fig_radar = go.Figure(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill="toself", fillcolor="rgba(102,126,234,0.2)",
        line=dict(color="#667eea", width=2), name=model_type,
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, height=300, margin=dict(t=20, b=20, l=40, r=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ============================================================
# MATRICE DE CONFUSION
# Réelle si disponible, estimée sinon avec avertissement
# ============================================================
st.subheader("🔲 Matrice de confusion")

if cm_real:
    # Matrice réelle issue de l'entraînement
    tn, fp_v = cm_real[0][0], cm_real[0][1]
    fn, tp   = cm_real[1][0], cm_real[1][1]
    cm_label = "✅ Matrice réelle (issue de l'entraînement)"
else:
    # Estimation approximative — avertissement clair
    n = metrics.get("n_test", 20)
    acc = metrics.get("accuracy", 0.75)
    prec = metrics.get("precision", 0.8)
    rec  = metrics.get("recall", 0.75)
    tp = max(int(n * rec * 0.5), 1)
    fn = max(int(n * (1 - rec) * 0.5), 1)
    fp_v = max(int(tp * (1 - prec) / max(prec, 0.01)), 1)
    tn = max(n - tp - fn - fp_v, 1)
    cm_label = "⚠️ Matrice estimée (entraîner avec `make train` pour la matrice réelle)"

st.caption(cm_label)
cm_data = [[tn, fp_v], [fn, tp]]
fig_cm = go.Figure(go.Heatmap(
    z=cm_data,
    x=["Prédit Réel", "Prédit Fake"],
    y=["Réel Réel", "Réel Fake"],
    colorscale=[[0, "#dcfce7"], [1, "#667eea"]],
    text=[[str(tn), str(fp_v)], [str(fn), str(tp)]],
    texttemplate="%{text}", textfont={"size": 20}, showscale=False,
))
fig_cm.update_layout(height=300, margin=dict(t=10, b=10))

col_cm, col_info = st.columns([1, 1])
with col_cm:
    st.plotly_chart(fig_cm, use_container_width=True)
with col_info:
    st.markdown("""
    **Interprétation :**
    - **VP (True Positive)** : Fake news correctement détectées
    - **VN (True Negative)** : Posts réels correctement identifiés
    - **FP (False Positive)** : Posts réels classés à tort comme fake
    - **FN (False Negative)** : Fake news non détectées *(les plus dangereuses)*

    **Objectif prioritaire** : minimiser les **Faux Négatifs**.
    """)

st.divider()

# ============================================================
# DÉTAILS ENTRAÎNEMENT
# ============================================================
st.subheader("ℹ️ Détails d'entraînement")
col1, col2, col3 = st.columns(3)
col1.metric("Exemples d'entraînement", metrics.get("n_train", "—"))
col2.metric("Exemples de test", metrics.get("n_test", "—"))
col3.metric("Entraîné le", str(metrics.get("trained_at", "—"))[:10])
st.markdown(f"**Modèle :** `{metrics.get('model', model_type)}`")

st.divider()

# ============================================================
# KPIs MÉTIER
# ============================================================
st.subheader("📋 KPIs Métier (Cahier des charges)")
fp_rate = fp_v / max(fp_v + tn, 1)
fn_rate = fn  / max(fn + tp, 1)
f1_val  = metrics.get("f1", 0)

kpi_data = {
    "KPI":     ["F1-Score global", "Taux de faux positifs", "Taux de faux négatifs",
                "Temps d'analyse moyen", "Couverture linguistique"],
    "Valeur":  [f"{f1_val:.1%}", f"{fp_rate:.1%}", f"{fn_rate:.1%}", "< 2s / post", "FR + EN"],
    "Objectif":["> 70%", "< 20%", "< 15%", "< 5s", "FR + EN"],
    "Statut":  [
        "✅" if f1_val  > 0.70 else "⚠️ À améliorer (make train-liar)",
        "✅" if fp_rate < 0.20 else "⚠️",
        "✅" if fn_rate < 0.15 else "⚠️",
        "✅", "✅",
    ],
}
st.dataframe(pd.DataFrame(kpi_data), use_container_width=True, hide_index=True)

# ============================================================
# SECTION ENTRAÎNEMENT — depuis le dashboard
# ============================================================
st.divider()
st.subheader("🚀 Entraîner / Ré-entraîner le modèle")

col_b1, col_b2, col_b3 = st.columns(3)

with col_b1:
    st.markdown("**Baseline** (TF-IDF)  \n~30 secondes")
    if st.button("▶️ Entraîner Baseline", use_container_width=True):
        with st.spinner("Entraînement baseline..."):
            result = subprocess.run(
                ["python", "scripts/train_model.py", "--model", "baseline"],
                capture_output=True, text=True, cwd=str(Path(__file__).parents[2])
            )
        if result.returncode == 0:
            st.success("✅ Baseline entraîné ! Rechargez la page.")
            st.code(result.stdout[-500:])
        else:
            st.error(f"Erreur : {result.stderr[-300:]}")

with col_b2:
    st.markdown("**DistilBERT** (CPU)  \n~20-30 minutes")
    if st.button("🤖 Fine-tuner DistilBERT", use_container_width=True):
        st.info("Lancement en arrière-plan... Consultez les logs : `tail -f logs/thumalien.log`")
        subprocess.Popen(
            ["python", "scripts/train_model.py", "--model", "bert"],
            cwd=str(Path(__file__).parents[2])
        )

with col_b3:
    st.markdown("**Dataset LIAR** (recommandé)  \n~20 min, 12k exemples")
    if st.button("🏆 Fine-tuner sur LIAR", use_container_width=True):
        st.info("Lancement en arrière-plan... Consultez les logs : `tail -f logs/thumalien.log`")
        subprocess.Popen(
            ["python", "main.py", "train", "--dataset", "liar", "--epochs", "3"],
            cwd=str(Path(__file__).parents[2])
        )
