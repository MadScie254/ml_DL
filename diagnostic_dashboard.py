# diagnostic_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# Load and train
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names
target_names = cancer.target_names
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
probs = clf.predict_proba(X)
preds = clf.predict(X)
confidences = np.max(probs, axis=1)

# DataFrame for plotting
df = pd.DataFrame(X, columns=feature_names)
df['actual'] = [target_names[i] for i in y]
df['predicted'] = [target_names[i] for i in preds]
df['confidence'] = confidences

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# App UI
st.set_page_config(layout="wide")
st.title("🧠 Ultra-Immersive Diagnostic Dashboard")

tabs = st.tabs(["📊 Patient Scan", "📈 PCA Explorer", "❌ Misclass Viewer", "🌡️ Heatmap"])

# ----------------------- Tab 1 ------------------------
with tabs[0]:
    st.header("📊 Patient Diagnostic Scan")
    idx = st.slider("Patient Index", 0, len(df)-1, 0)

    patient = df.iloc[idx]
    st.markdown(f"### Predicted: **{patient['predicted'].upper()}**")
    st.markdown(f"### Actual: **{patient['actual'].upper()}**")
    st.markdown(f"### Confidence: `{patient['confidence']:.2%}`")

    # Radar plot
    radar_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
    radar_values = [patient[f] for f in radar_features]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=radar_values,
        theta=radar_features,
        fill='toself',
        name='Patient Profile'
    ))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(radar_fig, use_container_width=True)

# ----------------------- Tab 2 ------------------------
with tabs[1]:
    st.header("📈 PCA Explorer with Prediction Clusters")
    fig = px.scatter(
        df, x='PCA1', y='PCA2',
        color='predicted',
        symbol='actual',
        size='confidence',
        hover_data=['confidence'],
        title="🧬 PCA Projection + Class Labels",
        height=650
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------- Tab 3 ------------------------
with tabs[2]:
    st.header("❌ Misclassification Inspector")
    misclassified = df[df['predicted'] != df['actual']]
    st.write(f"🔎 Total Misclassifications: {len(misclassified)}")

    fig = px.scatter(
        misclassified,
        x='mean concavity',
        y='mean smoothness',
        color='confidence',
        hover_data=['predicted', 'actual'],
        size='mean perimeter',
        title="Where the Model Fails",
        color_continuous_scale='OrRd',
        height=650
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------- Tab 4 ------------------------
with tabs[3]:
    st.header("🌡️ Confidence Heatmap in PCA Space")
    fig = px.scatter(
        df, x='PCA1', y='PCA2',
        color='confidence',
        color_continuous_scale='Turbo',
        symbol='predicted',
        hover_data=['predicted', 'actual'],
        title="Trust Landscape of the Model",
        height=650
    )
    st.plotly_chart(fig, use_container_width=True)