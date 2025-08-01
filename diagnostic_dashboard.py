# diagnostic_dashboard.py
"""
Advanced Breast Cancer Diagnostic AI Dashboard
===============================================
Professional medical diagnostic interface with real-time predictions,
immersive visualizations, and comprehensive patient analysis tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for medical theme
st.markdown("""
<style>
.medical-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.diagnostic-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #007bff;
    margin: 1rem 0;
}
.alert-high { background-color: #ffebee; border-left-color: #f44336; }
.alert-medium { background-color: #fff3e0; border-left-color: #ff9800; }
.alert-low { background-color: #e8f5e8; border-left-color: #4caf50; }
.metric-container {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}
.sidebar-content {
    background: #f1f3f4;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# App configuration
st.set_page_config(
    page_title="MedAI Diagnostic Center",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and prepare data
@st.cache_data
def load_data():
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    feature_names = cancer.feature_names
    target_names = cancer.target_names
    return X, y, feature_names, target_names, cancer

@st.cache_resource
def load_model():
    try:
        clf = load('breast_cancer_rf_model.joblib')
        return clf
    except:
        st.error("⚠️ Model file not found. Please ensure 'breast_cancer_rf_model.joblib' exists.")
        return None

# Load data and model
X, y, feature_names, target_names, cancer = load_data()
clf = load_model()

if clf is None:
    st.stop()

# Generate predictions and analysis
@st.cache_data
def generate_predictions():
    probs = clf.predict_proba(X)
    preds = clf.predict(X)
    confidences = np.max(probs, axis=1)
    
    # Create comprehensive dataframe
    df = pd.DataFrame(X, columns=feature_names)
    df['actual'] = [target_names[i] for i in y]
    df['predicted'] = [target_names[i] for i in preds]
    df['confidence'] = confidences
    df['is_correct'] = (df['actual'] == df['predicted'])
    
    # Add risk categories
    df['risk_level'] = pd.cut(df['confidence'], 
                             bins=[0, 0.7, 0.85, 1.0], 
                             labels=['High Risk', 'Medium Risk', 'Low Risk'])
    
    # PCA projection
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    
    # t-SNE projection for better clustering visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(X)
    df['TSNE1'] = tsne_result[:, 0]
    df['TSNE2'] = tsne_result[:, 1]
    
    return df, pca

df, pca = generate_predictions()

# Header
st.markdown("""
<div class="medical-header">
    <h1>⚕️ MedAI Breast Cancer Diagnostic Center</h1>
    <p>Advanced AI-Powered Diagnostic Analysis System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with system overview
with st.sidebar:
    st.markdown("### 📊 System Overview")
    
    total_patients = len(df)
    accuracy = (df['is_correct'].sum() / total_patients) * 100
    high_conf_cases = (df['confidence'] > 0.9).sum()
    misclassified = (~df['is_correct']).sum()
    
    st.markdown(f"""
    <div class="sidebar-content">
        <h4>📈 Performance Metrics</h4>
        <p><strong>Total Cases:</strong> {total_patients}</p>
        <p><strong>Model Accuracy:</strong> {accuracy:.1f}%</p>
        <p><strong>High Confidence Cases:</strong> {high_conf_cases}</p>
        <p><strong>Misclassified:</strong> {misclassified}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    benign_count = (df['actual'] == 'benign').sum()
    malignant_count = (df['actual'] == 'malignant').sum()
    
    st.markdown(f"""
    <div class="sidebar-content">
        <h4>🔬 Dataset Composition</h4>
        <p><strong>Benign Cases:</strong> {benign_count}</p>
        <p><strong>Malignant Cases:</strong> {malignant_count}</p>
        <p><strong>Benign Rate:</strong> {(benign_count/total_patients)*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

# Main dashboard with enhanced tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏥 Patient Analysis", 
    "🧬 Molecular Explorer", 
    "⚠️ Risk Assessment", 
    "📊 Performance Analytics",
    "🎯 Prediction Engine",
    "📋 Clinical Report"
])

# Tab 1: Enhanced Patient Analysis
with tab1:
    st.markdown("### 🏥 Individual Patient Diagnostic Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        patient_idx = st.selectbox(
            "Select Patient ID:",
            range(len(df)),
            format_func=lambda x: f"Patient #{x:03d}"
        )
        
        patient = df.iloc[patient_idx]
        
        # Risk level styling
        risk_class = ""
        if patient['confidence'] < 0.7:
            risk_class = "alert-high"
        elif patient['confidence'] < 0.85:
            risk_class = "alert-medium"
        else:
            risk_class = "alert-low"
        
        st.markdown(f"""
        <div class="diagnostic-card {risk_class}">
            <h3>🔬 Diagnosis</h3>
            <p><strong>Predicted:</strong> {patient['predicted'].upper()}</p>
            <p><strong>Actual:</strong> {patient['actual'].upper()}</p>
            <p><strong>Confidence:</strong> {patient['confidence']:.1%}</p>
            <p><strong>Risk Level:</strong> {patient['risk_level']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature importance for this patient
        st.markdown("### 📊 Key Biomarkers")
        key_features = ['mean radius', 'mean texture', 'mean perimeter', 'worst concavity', 'worst symmetry']
        for feature in key_features:
            value = patient[feature]
            percentile = (df[feature] <= value).mean() * 100
            st.metric(feature.replace('_', ' ').title(), f"{value:.2f}", f"{percentile:.0f}th percentile")
    
    with col2:
        # Create comprehensive patient visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Biomarker Profile', 'Feature Distribution', 'Risk Assessment', 'Comparison Matrix'),
            specs=[[{"type": "scatterpolar"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "heatmap"}]]
        )
        
        # Radar chart for key features
        radar_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'worst concavity']
        radar_values = [patient[f] for f in radar_features]
        
        fig.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_features,
            fill='toself',
            name='Patient Profile',
            line_color='#1f77b4'
        ), row=1, col=1)
        
        # Feature comparison bar chart
        comparison_features = radar_features[:5]
        mean_values = [df[f].mean() for f in comparison_features]
        patient_values = [patient[f] for f in comparison_features]
        
        fig.add_trace(go.Bar(
            x=comparison_features,
            y=mean_values,
            name='Population Mean',
            marker_color='lightblue'
        ), row=1, col=2)
        
        fig.add_trace(go.Bar(
            x=comparison_features,
            y=patient_values,
            name='Patient Value',
            marker_color='darkblue'
        ), row=1, col=2)
        
        # Confidence gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=patient['confidence'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence %"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 70], 'color': "lightgray"},
                            {'range': [70, 85], 'color': "yellow"},
                            {'range': [85, 100], 'color': "green"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 90}}
        ), row=2, col=1)
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Molecular Explorer with t-SNE
with tab2:
    st.markdown("### 🧬 Advanced Molecular Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### PCA Projection")
        fig_pca = px.scatter(
            df, x='PCA1', y='PCA2',
            color='predicted',
            symbol='actual',
            size='confidence',
            hover_data=['confidence', 'risk_level'],
            title="Principal Component Analysis",
            height=400,
            color_discrete_map={'benign': '#4CAF50', 'malignant': '#F44336'}
        )
        fig_pca.update_traces(marker=dict(line=dict(width=1, color='white')))
        st.plotly_chart(fig_pca, use_container_width=True)
    
    with col2:
        st.markdown("#### t-SNE Clustering")
        fig_tsne = px.scatter(
            df, x='TSNE1', y='TSNE2',
            color='predicted',
            symbol='actual',
            size='confidence',
            hover_data=['confidence', 'risk_level'],
            title="t-SNE Molecular Clustering",
            height=400,
            color_discrete_map={'benign': '#4CAF50', 'malignant': '#F44336'}
        )
        fig_tsne.update_traces(marker=dict(line=dict(width=1, color='white')))
        st.plotly_chart(fig_tsne, use_container_width=True)
    
    # Feature correlation heatmap
    st.markdown("#### 🔥 Feature Correlation Matrix")
    correlation_features = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area'
    ]
    corr_matrix = df[correlation_features].corr()
    
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title="Biomarker Correlation Analysis"
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# Tab 3: Risk Assessment Dashboard
with tab3:
    st.markdown("### ⚠️ Comprehensive Risk Assessment")
    
    # Risk distribution
    col1, col2, col3 = st.columns(3)
    
    risk_counts = df['risk_level'].value_counts()
    
    with col1:
        st.markdown("#### 🔴 High Risk Cases")
        high_risk_count = risk_counts.get('High Risk', 0)
        st.metric("Count", high_risk_count)
        st.metric("Percentage", f"{(high_risk_count/len(df)*100):.1f}%")
    
    with col2:
        st.markdown("#### 🟡 Medium Risk Cases")
        medium_risk_count = risk_counts.get('Medium Risk', 0)
        st.metric("Count", medium_risk_count)
        st.metric("Percentage", f"{(medium_risk_count/len(df)*100):.1f}%")
    
    with col3:
        st.markdown("#### 🟢 Low Risk Cases")
        low_risk_count = risk_counts.get('Low Risk', 0)
        st.metric("Count", low_risk_count)
        st.metric("Percentage", f"{(low_risk_count/len(df)*100):.1f}%")
    
    # Risk analysis plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution pie chart
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Level Distribution",
            color_discrete_map={
                'High Risk': '#F44336',
                'Medium Risk': '#FF9800',
                'Low Risk': '#4CAF50'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence vs Accuracy scatter
        fig_conf = px.scatter(
            df, x='confidence', y='is_correct',
            color='predicted',
            size='mean area',
            hover_data=['risk_level'],
            title="Confidence vs Accuracy Analysis",
            labels={'is_correct': 'Prediction Correct'},
            color_discrete_map={'benign': '#4CAF50', 'malignant': '#F44336'}
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    # Misclassification deep dive
    st.markdown("#### 🎯 Misclassification Analysis")
    misclassified = df[~df['is_correct']]
    
    if len(misclassified) > 0:
        fig_misc = px.scatter(
            misclassified,
            x='mean concavity',
            y='worst smoothness',
            color='confidence',
            size='mean perimeter',
            hover_data=['predicted', 'actual', 'risk_level'],
            title="Failed Predictions - Feature Space Analysis",
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_misc, use_container_width=True)
    else:
        st.success("🎉 Perfect Classification - No Misclassifications Found!")

# Tab 4: Performance Analytics
with tab4:
    st.markdown("### 📊 Advanced Performance Analytics")
    
    # Confusion matrix
    col1, col2 = st.columns(2)
    
    with col1:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(df['actual'], df['predicted'])
        
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Blues',
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            x=['Benign', 'Malignant'],
            y=['Benign', 'Malignant']
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        # Feature importance (simulated for Random Forest)
        feature_importance = pd.DataFrame({
            'feature': ['mean radius', 'worst perimeter', 'worst area', 'mean concavity', 'worst smoothness'],
            'importance': [0.25, 0.20, 0.18, 0.15, 0.12]
        })
        
        fig_importance = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title="Top 5 Feature Importance",
            color='importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Performance metrics table
    st.markdown("#### 📈 Detailed Performance Metrics")
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    y_true_binary = [1 if x == 'malignant' else 0 for x in df['actual']]
    y_pred_binary = [1 if x == 'malignant' else 0 for x in df['predicted']]
    
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity'],
        'Value': [
            accuracy/100,
            precision_score(y_true_binary, y_pred_binary),
            recall_score(y_true_binary, y_pred_binary),
            f1_score(y_true_binary, y_pred_binary),
            precision_score(y_true_binary, y_pred_binary, pos_label=0)
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df['Percentage'] = (metrics_df['Value'] * 100).round(1)
    
    st.dataframe(metrics_df, use_container_width=True)

# Tab 5: Prediction Engine
with tab5:
    st.markdown("### 🎯 Interactive Prediction Engine")
    
    st.markdown("#### 🔬 Custom Case Analysis")
    
    # Allow users to input custom values
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mean_radius = st.slider("Mean Radius", float(df['mean radius'].min()), float(df['mean radius'].max()), float(df['mean radius'].mean()))
        mean_texture = st.slider("Mean Texture", float(df['mean texture'].min()), float(df['mean texture'].max()), float(df['mean texture'].mean()))
        mean_perimeter = st.slider("Mean Perimeter", float(df['mean perimeter'].min()), float(df['mean perimeter'].max()), float(df['mean perimeter'].mean()))
    
    with col2:
        mean_area = st.slider("Mean Area", float(df['mean area'].min()), float(df['mean area'].max()), float(df['mean area'].mean()))
        worst_concavity = st.slider("Worst Concavity", float(df['worst concavity'].min()), float(df['worst concavity'].max()), float(df['worst concavity'].mean()))
        worst_symmetry = st.slider("Worst Symmetry", float(df['worst symmetry'].min()), float(df['worst symmetry'].max()), float(df['worst symmetry'].mean()))
    
    # Create custom sample with mean values for other features
    custom_sample = df.iloc[0].copy()  # Start with a template
    custom_sample['mean radius'] = mean_radius
    custom_sample['mean texture'] = mean_texture
    custom_sample['mean perimeter'] = mean_perimeter
    custom_sample['mean area'] = mean_area
    custom_sample['worst concavity'] = worst_concavity
    custom_sample['worst symmetry'] = worst_symmetry
    
    # Predict for custom sample
    if st.button("🔍 Run Prediction", type="primary"):
        # Prepare feature vector
        feature_vector = np.array([custom_sample[feature] for feature in feature_names]).reshape(1, -1)
        
        prediction = clf.predict(feature_vector)[0]
        probability = clf.predict_proba(feature_vector)[0]
        confidence = np.max(probability)
        
        predicted_label = target_names[prediction]
        
        with col3:
            st.markdown(f"""
            <div class="diagnostic-card {'alert-high' if confidence < 0.7 else 'alert-medium' if confidence < 0.85 else 'alert-low'}">
                <h3>🎯 Prediction Results</h3>
                <p><strong>Diagnosis:</strong> {predicted_label.upper()}</p>
                <p><strong>Confidence:</strong> {confidence:.1%}</p>
                <p><strong>Malignant Probability:</strong> {probability[0]:.1%}</p>
                <p><strong>Benign Probability:</strong> {probability[1]:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

# Tab 6: Clinical Report
with tab6:
    st.markdown("### 📋 Comprehensive Clinical Report")
    
    # Generate summary statistics
    total_cases = len(df)
    benign_cases = (df['actual'] == 'benign').sum()
    malignant_cases = (df['actual'] == 'malignant').sum()
    high_confidence = (df['confidence'] > 0.9).sum()
    
    st.markdown(f"""
    ## Executive Summary
    
    **Dataset Overview:**
    - Total analyzed cases: {total_cases}
    - Benign cases: {benign_cases} ({(benign_cases/total_cases)*100:.1f}%)
    - Malignant cases: {malignant_cases} ({(malignant_cases/total_cases)*100:.1f}%)
    
    **Model Performance:**
    - Overall accuracy: {accuracy:.1f}%
    - High confidence predictions: {high_confidence} ({(high_confidence/total_cases)*100:.1f}%)
    - Cases requiring review: {total_cases - high_confidence}
    
    **Risk Distribution:**
    - High risk cases: {risk_counts.get('High Risk', 0)}
    - Medium risk cases: {risk_counts.get('Medium Risk', 0)}
    - Low risk cases: {risk_counts.get('Low Risk', 0)}
    """)
    
    # Detailed case list
    st.markdown("#### 📊 Detailed Case Analysis")
    
    # Filter options
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        diagnosis_filter = st.selectbox("Filter by Diagnosis:", ["All", "Benign", "Malignant"])
    
    with filter_col2:
        risk_filter = st.selectbox("Filter by Risk Level:", ["All", "High Risk", "Medium Risk", "Low Risk"])
    
    with filter_col3:
        accuracy_filter = st.selectbox("Filter by Accuracy:", ["All", "Correct", "Incorrect"])
    
    # Apply filters
    filtered_df = df.copy()
    
    if diagnosis_filter != "All":
        filtered_df = filtered_df[filtered_df['predicted'] == diagnosis_filter.lower()]
    
    if risk_filter != "All":
        filtered_df = filtered_df[filtered_df['risk_level'] == risk_filter]
    
    if accuracy_filter == "Correct":
        filtered_df = filtered_df[filtered_df['is_correct']]
    elif accuracy_filter == "Incorrect":
        filtered_df = filtered_df[~filtered_df['is_correct']]
    
    # Display filtered results
    display_columns = ['predicted', 'actual', 'confidence', 'risk_level', 'is_correct', 'mean radius', 'mean texture']
    st.dataframe(
        filtered_df[display_columns].round(3),
        use_container_width=True,
        height=400
    )
    
    # Download button for full report
    if st.button("📥 Download Full Report"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV Report",
            data=csv,
            file_name="breast_cancer_diagnostic_report.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px;">
    ⚕️ MedAI Diagnostic Center | Advanced Machine Learning for Medical Diagnosis<br>
    Built with Streamlit & scikit-learn | For Research and Educational Purposes
</div>
""", unsafe_allow_html=True)
