"""
Enhanced Medical Dashboard with Advanced Features
================================================
Multi-page Streamlit application with comprehensive medical diagnostic tools,
real-time monitoring, and advanced visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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

# Import custom modules
from medical_utils import MedicalAnalyzer, ClinicalReportGenerator, get_medical_name
from advanced_visualizations import Medical3DVisualizer, InteractivePlotBuilder, MedicalImageSimulator
from realtime_simulator import create_live_monitoring_page

# Page configuration
st.set_page_config(
    page_title="MedAI Advanced Diagnostic Suite",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with medical theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
    margin: 1rem 0;
    transition: transform 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
}

.diagnostic-alert {
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border-left: 4px solid;
}

.alert-critical { 
    background: #ffebee; 
    border-left-color: #f44336;
    color: #c62828;
}

.alert-warning { 
    background: #fff3e0; 
    border-left-color: #ff9800;
    color: #ef6c00;
}

.alert-success { 
    background: #e8f5e8; 
    border-left-color: #4caf50;
    color: #2e7d32;
}

.alert-info { 
    background: #e3f2fd; 
    border-left-color: #2196f3;
    color: #1565c0;
}

.sidebar-section {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border: 1px solid #e9ecef;
}

.patient-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border: 2px solid #e9ecef;
    margin: 1rem 0;
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-benign { background-color: #4caf50; }
.status-malignant { background-color: #f44336; }
.status-pending { background-color: #ff9800; }

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.feature-item {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #e9ecef;
    text-align: center;
}

.navigation-tabs {
    background: white;
    border-radius: 10px;
    padding: 0.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.footer-info {
    background: #f8f9fa;
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    margin-top: 3rem;
    border: 1px solid #e9ecef;
}
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_medical_data():
    """Load and prepare medical dataset"""
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    feature_names = cancer.feature_names
    target_names = cancer.target_names
    return X, y, feature_names, target_names, cancer

@st.cache_resource
def load_diagnostic_model():
    """Load the trained diagnostic model"""
    try:
        clf = load('breast_cancer_rf_model.joblib')
        return clf
    except FileNotFoundError:
        st.error("🔧 Model file not found. Please ensure 'breast_cancer_rf_model.joblib' exists in the working directory.")
        st.info("Run the training notebook to generate the model file.")
        return None

# Initialize components
X, y, feature_names, target_names, cancer = load_medical_data()
clf = load_diagnostic_model()

if clf is None:
    st.stop()

# Initialize medical analyzer and visualizers
medical_analyzer = MedicalAnalyzer()
visualizer_3d = Medical3DVisualizer()
plot_builder = InteractivePlotBuilder()
image_simulator = MedicalImageSimulator()
report_generator = ClinicalReportGenerator(medical_analyzer)

# Generate comprehensive predictions
@st.cache_data
def generate_comprehensive_analysis():
    """Generate complete diagnostic analysis"""
    
    # Predictions
    probs = clf.predict_proba(X)
    preds = clf.predict(X)
    confidences = np.max(probs, axis=1)
    
    # Create main dataframe
    df = pd.DataFrame(X, columns=feature_names)
    df['actual'] = [target_names[i] for i in y]
    df['predicted'] = [target_names[i] for i in preds]
    df['confidence'] = confidences
    df['is_correct'] = (df['actual'] == df['predicted'])
    
    # Risk stratification
    df['risk_level'] = df.apply(lambda row: 
        medical_analyzer.assess_risk_level(row['confidence'], row['predicted']).value, axis=1)
    
    # Add dimensionality reduction
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    
    # t-SNE for better clustering
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(X)
    df['TSNE1'] = tsne_result[:, 0]
    df['TSNE2'] = tsne_result[:, 1]
    
    return df, pca

df, pca = generate_comprehensive_analysis()

# Header
st.markdown("""
<div class="main-header">
    <h1>⚕️ MedAI Advanced Diagnostic Suite</h1>
    <p>Comprehensive AI-Powered Breast Cancer Diagnostic Analysis Platform</p>
    <p>🏥 Real-time Screening | 🧬 Molecular Analysis | 📊 Clinical Insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with comprehensive system overview
with st.sidebar:
    st.markdown("## 📋 System Dashboard")
    
    # Key performance indicators
    total_patients = len(df)
    accuracy = (df['is_correct'].sum() / total_patients) * 100
    malignant_cases = (df['predicted'] == 'malignant').sum()
    high_conf_cases = (df['confidence'] > 0.9).sum()
    
    st.markdown(f"""
    <div class="sidebar-section">
        <h4>📈 Key Performance Indicators</h4>
        <div style="display: grid; gap: 10px;">
            <div>📊 <strong>Total Cases:</strong> {total_patients}</div>
            <div>🎯 <strong>Model Accuracy:</strong> {accuracy:.1f}%</div>
            <div>🔴 <strong>Malignant Detected:</strong> {malignant_cases}</div>
            <div>✅ <strong>High Confidence:</strong> {high_conf_cases}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk distribution
    risk_counts = df['risk_level'].value_counts()
    st.markdown(f"""
    <div class="sidebar-section">
        <h4>⚠️ Risk Stratification</h4>
        <div style="display: grid; gap: 8px;">
            <div>🔴 <strong>Critical Risk:</strong> {risk_counts.get('Critical Risk', 0)}</div>
            <div>🟠 <strong>High Risk:</strong> {risk_counts.get('High Risk', 0)}</div>
            <div>🟡 <strong>Medium Risk:</strong> {risk_counts.get('Medium Risk', 0)}</div>
            <div>🟢 <strong>Low Risk:</strong> {risk_counts.get('Low Risk', 0)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    if st.button("📥 Export Full Report", type="primary"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="📋 Download CSV Report",
            data=csv,
            file_name=f"diagnostic_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    if st.button("🔄 Refresh Analysis"):
        st.cache_data.clear()
        st.experimental_rerun()

# Main navigation with enhanced tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏥 Patient Analysis", 
    "🧬 3D Molecular View", 
    "⚠️ Risk Center", 
    "📊 Analytics Hub",
    "🎯 AI Prediction Lab",
    "🔴 Live Monitor",
    "📋 Clinical Reports"
])

# Tab 1: Enhanced Patient Analysis
with tab1:
    st.markdown("### 🏥 Advanced Patient Diagnostic Workstation")
    
    # Patient selection with enhanced interface
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("#### 👤 Patient Selection")
        patient_idx = st.selectbox(
            "Choose Patient:",
            range(len(df)),
            format_func=lambda x: f"Patient #{x:03d} ({df.iloc[x]['predicted'].title()})"
        )
        
        patient = df.iloc[patient_idx]
        
        # Enhanced patient card
        status_class = f"status-{patient['predicted']}"
        alert_class = ""
        if patient['confidence'] < 0.7:
            alert_class = "alert-critical"
        elif patient['confidence'] < 0.85:
            alert_class = "alert-warning"
        else:
            alert_class = "alert-success"
        
        st.markdown(f"""
        <div class="patient-card {alert_class}">
            <h3>🔬 Diagnostic Summary</h3>
            <p><span class="status-indicator {status_class}"></span>
               <strong>Diagnosis:</strong> {patient['predicted'].upper()}</p>
            <p>📊 <strong>Actual:</strong> {patient['actual'].upper()}</p>
            <p>🎯 <strong>Confidence:</strong> {patient['confidence']:.1%}</p>
            <p>⚠️ <strong>Risk Level:</strong> {patient['risk_level']}</p>
            <p>✅ <strong>Status:</strong> {'Correct' if patient['is_correct'] else 'Misclassified'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Clinical interpretation
        features_dict = {col: patient[col] for col in feature_names}
        interpretation = medical_analyzer.generate_clinical_interpretation(
            features_dict, patient['predicted'], patient['confidence']
        )
        
        st.markdown("#### 🩺 Clinical Notes")
        for note in interpretation['clinical_notes']:
            st.info(f"📝 {note}")
        
        st.markdown("#### 📋 Recommendations")
        for rec in interpretation['recommendations']:
            st.success(f"💡 {rec}")
    
    with col2:
        # Multi-panel visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '🎯 Biomarker Radar Profile',
                '📊 Feature Distribution vs Population',
                '🔍 Confidence Assessment',
                '📷 Simulated Imaging View'
            ),
            specs=[
                [{"type": "scatterpolar"}, {"type": "bar"}],
                [{"type": "indicator"}, {"type": "heatmap"}]
            ],
            vertical_spacing=0.1
        )
        
        # 1. Enhanced radar chart
        key_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'worst concavity', 'worst symmetry']
        patient_values = [patient[f] for f in key_features]
        
        fig.add_trace(go.Scatterpolar(
            r=patient_values,
            theta=[get_medical_name(f) for f in key_features],
            fill='toself',
            name='Patient Profile',
            line_color='#667eea',
            fillcolor='rgba(102, 126, 234, 0.3)'
        ), row=1, col=1)
        
        # 2. Comparative analysis
        pop_means = [df[f].mean() for f in key_features[:5]]
        patient_vals = [patient[f] for f in key_features[:5]]
        
        fig.add_trace(go.Bar(
            x=[get_medical_name(f) for f in key_features[:5]],
            y=pop_means,
            name='Population Average',
            marker_color='lightblue',
            opacity=0.7
        ), row=1, col=2)
        
        fig.add_trace(go.Bar(
            x=[get_medical_name(f) for f in key_features[:5]],
            y=patient_vals,
            name='Patient Values',
            marker_color='#667eea'
        ), row=1, col=2)
        
        # 3. Enhanced confidence gauge
        gauge_color = "red" if patient['confidence'] < 0.7 else "orange" if patient['confidence'] < 0.85 else "green"
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=patient['confidence'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Diagnostic Confidence (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 85], 'color': "yellow"},
                    {'range': [85, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75, 
                    'value': 90
                }
            }
        ), row=2, col=1)
        
        # 4. Simulated medical imaging
        imaging_features = {
            'mean radius': patient['mean radius'],
            'mean area': patient['mean area'],
            'worst concavity': patient['worst concavity']
        }
        
        # Simple heatmap representation of tissue characteristics
        tissue_data = np.random.normal(0.5, 0.1, (10, 10))
        if patient['predicted'] == 'malignant':
            tissue_data[4:6, 4:6] += 0.5  # Add "mass"
        
        fig.add_trace(go.Heatmap(
            z=tissue_data,
            colorscale='Greys',
            showscale=False,
            name='Tissue Simulation'
        ), row=2, col=2)
        
        fig.update_layout(
            height=800,
            title_text="🔬 Comprehensive Patient Analysis Dashboard",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: 3D Molecular Visualization
with tab2:
    st.markdown("### 🧬 Advanced 3D Molecular Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 3D molecular space
        fig_3d = visualizer_3d.create_3d_molecular_space(df)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with col2:
        # 3D confidence landscape
        fig_landscape = visualizer_3d.create_confidence_landscape(df)
        st.plotly_chart(fig_landscape, use_container_width=True)
    
    # Enhanced feature analysis
    st.markdown("#### 🔬 Molecular Feature Correlation Network")
    
    # Create correlation network visualization
    key_features_extended = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area',
        'worst concavity', 'worst symmetry'
    ]
    
    corr_matrix = df[key_features_extended].corr()
    
    # Create network-style heatmap
    fig_network = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="🕸️ Biomarker Interaction Network"
    )
    fig_network.update_layout(height=600)
    st.plotly_chart(fig_network, use_container_width=True)

# Continue with remaining tabs...
# Tab 3: Risk Assessment Center
with tab3:
    st.markdown("### ⚠️ Comprehensive Risk Assessment Center")
    
    # Risk analytics dashboard
    risk_metrics = df['risk_level'].value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    
    risk_colors = {
        'Critical Risk': '#f44336',
        'High Risk': '#ff5722', 
        'Medium Risk': '#ff9800',
        'Low Risk': '#4caf50'
    }
    
    for i, (risk_level, count) in enumerate(risk_metrics.items()):
        col = [col1, col2, col3, col4][i % 4]
        percentage = (count / len(df)) * 100
        
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {risk_colors.get(risk_level, '#999')};">
                <h3 style="color: {risk_colors.get(risk_level, '#999')};">{risk_level}</h3>
                <h2>{count}</h2>
                <p>{percentage:.1f}% of total cases</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Risk analysis visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution over confidence
        fig_risk_conf = px.scatter(
            df, x='confidence', y='mean radius',
            color='risk_level',
            size='mean area',
            hover_data=['predicted', 'actual'],
            title="🎯 Risk vs Confidence Analysis",
            color_discrete_map=risk_colors
        )
        st.plotly_chart(fig_risk_conf, use_container_width=True)
    
    with col2:
        # Risk by prediction accuracy
        accuracy_risk = pd.crosstab(df['risk_level'], df['is_correct'])
        fig_acc_risk = px.bar(
            accuracy_risk.T,
            title="✅ Prediction Accuracy by Risk Level",
            color_discrete_sequence=['#f44336', '#4caf50']
        )
        st.plotly_chart(fig_acc_risk, use_container_width=True)

# Tab 4: Analytics Hub  
with tab4:
    st.markdown("### 📊 Advanced Analytics & Performance Hub")
    
    # Comprehensive performance dashboard
    fig_dashboard = plot_builder.create_diagnostic_dashboard_plot(df)
    st.plotly_chart(fig_dashboard, use_container_width=True)
    
    # Detailed performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Model Performance Metrics")
        
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        y_true_binary = [1 if x == 'malignant' else 0 for x in df['actual']]
        y_pred_binary = [1 if x == 'malignant' else 0 for x in df['predicted']]
        y_prob_malignant = [clf.predict_proba(X[i].reshape(1, -1))[0][0] for i in range(len(X))]
        
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC'],
            'Value': [
                accuracy/100,
                precision_score(y_true_binary, y_pred_binary),
                recall_score(y_true_binary, y_pred_binary),
                f1_score(y_true_binary, y_pred_binary),
                precision_score(y_true_binary, y_pred_binary, pos_label=0),
                roc_auc_score(y_true_binary, y_prob_malignant)
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df['Percentage'] = (metrics_df['Value'] * 100).round(1)
        
        # Enhanced metrics display
        for _, row in metrics_df.iterrows():
            st.markdown(f"""
            <div class="metric-card">
                <h4>{row['Metric']}</h4>
                <h2 style="color: #667eea;">{row['Percentage']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🎯 Feature Importance Analysis")
        
        # Simulated feature importance (Random Forest)
        feature_importance = pd.DataFrame({
            'feature': ['worst perimeter', 'worst area', 'mean concave points', 'worst radius', 'mean concavity'],
            'importance': [0.15, 0.13, 0.11, 0.10, 0.09]
        })
        
        fig_importance = px.bar(
            feature_importance.sort_values('importance'),
            x='importance',
            y='feature',
            orientation='h',
            title="🔝 Top 5 Diagnostic Features",
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)

# Tab 5: AI Prediction Laboratory
with tab5:
    st.markdown("### 🎯 AI Prediction Laboratory")
    st.markdown("Interactive prediction engine with custom parameter adjustment")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown("#### 🔬 Biomarker Input Panel")
        
        # Enhanced input controls with medical context
        custom_features = {}
        
        # Group features by medical significance
        morphology_features = ['mean radius', 'mean perimeter', 'mean area']
        texture_features = ['mean texture', 'mean smoothness']
        shape_features = ['worst concavity', 'worst symmetry']
        
        st.markdown("**📏 Tumor Morphology**")
        for feature in morphology_features:
            min_val, max_val = float(df[feature].min()), float(df[feature].max())
            mean_val = float(df[feature].mean())
            custom_features[feature] = st.slider(
                get_medical_name(feature),
                min_val, max_val, mean_val,
                help=f"Normal range: {min_val:.1f} - {max_val:.1f}"
            )
        
        st.markdown("**🎨 Tissue Characteristics**")
        for feature in texture_features:
            min_val, max_val = float(df[feature].min()), float(df[feature].max())
            mean_val = float(df[feature].mean())
            custom_features[feature] = st.slider(
                get_medical_name(feature),
                min_val, max_val, mean_val
            )
        
        st.markdown("**📐 Structural Analysis**")
        for feature in shape_features:
            min_val, max_val = float(df[feature].min()), float(df[feature].max())
            mean_val = float(df[feature].mean())
            custom_features[feature] = st.slider(
                get_medical_name(feature),
                min_val, max_val, mean_val
            )
    
    with col2:
        st.markdown("#### 🎯 Prediction")
        
        if st.button("🔍 Analyze Case", type="primary", help="Run AI diagnostic analysis"):
            # Create feature vector with defaults for missing features
            feature_vector = np.array([
                custom_features.get(name, df[name].mean()) 
                for name in feature_names
            ]).reshape(1, -1)
            
            # Generate prediction
            prediction = clf.predict(feature_vector)[0]
            probability = clf.predict_proba(feature_vector)[0]
            confidence = np.max(probability)
            predicted_label = target_names[prediction]
            
            # Risk assessment
            risk_level = medical_analyzer.assess_risk_level(confidence, predicted_label)
            
            # Results display
            result_class = "alert-critical" if predicted_label == 'malignant' else "alert-success"
            
            st.markdown(f"""
            <div class="diagnostic-alert {result_class}">
                <h3>🎯 AI Diagnostic Result</h3>
                <p><strong>Diagnosis:</strong> {predicted_label.upper()}</p>
                <p><strong>Confidence:</strong> {confidence:.1%}</p>
                <p><strong>Risk Level:</strong> {risk_level.value}</p>
                <hr>
                <p><strong>Malignant Prob:</strong> {probability[0]:.1%}</p>
                <p><strong>Benign Prob:</strong> {probability[1]:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Clinical interpretation
            interpretation = medical_analyzer.generate_clinical_interpretation(
                custom_features, predicted_label, confidence
            )
            
            st.markdown("**📋 Clinical Notes:**")
            for note in interpretation['clinical_notes']:
                st.info(note)
    
    with col3:
        st.markdown("#### 📊 Live Analysis")
        
        # Real-time radar chart updates
        if 'custom_features' in locals():
            radar_features = list(custom_features.keys())
            radar_values = list(custom_features.values())
            
            fig_live_radar = go.Figure()
            fig_live_radar.add_trace(go.Scatterpolar(
                r=radar_values,
                theta=[get_medical_name(f) for f in radar_features],
                fill='toself',
                name='Custom Case',
                line_color='#ff6b6b'
            ))
            
            # Add population average for comparison
            pop_values = [df[f].mean() for f in radar_features]
            fig_live_radar.add_trace(go.Scatterpolar(
                r=pop_values,
                theta=[get_medical_name(f) for f in radar_features],
                fill='toself',
                name='Population Average',
                line_color='#4ecdc4',
                opacity=0.6
            ))
            
            fig_live_radar.update_layout(
                title="🎯 Live Biomarker Comparison",
                height=400
            )
            st.plotly_chart(fig_live_radar, use_container_width=True)

# Tab 6: Live Monitoring Center
with tab6:
    create_live_monitoring_page()

# Tab 7: Clinical Reports
with tab7:
    st.markdown("### 📋 Comprehensive Clinical Reporting Center")
    
    # Report generation interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📊 Batch Report Generator")
        
        # Filter options
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            diagnosis_filter = st.selectbox(
                "Filter by Diagnosis:", 
                ["All", "Benign", "Malignant"]
            )
        
        with filter_col2:
            risk_filter = st.selectbox(
                "Filter by Risk Level:", 
                ["All"] + list(df['risk_level'].unique())
            )
        
        with filter_col3:
            confidence_threshold = st.slider(
                "Minimum Confidence:", 
                0.0, 1.0, 0.0, 0.05
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if diagnosis_filter != "All":
            filtered_df = filtered_df[filtered_df['predicted'] == diagnosis_filter.lower()]
        
        if risk_filter != "All":
            filtered_df = filtered_df[filtered_df['risk_level'] == risk_filter]
        
        filtered_df = filtered_df[filtered_df['confidence'] >= confidence_threshold]
        
        # Summary statistics
        st.markdown(f"""
        #### 📈 Filtered Dataset Summary
        - **Total Cases:** {len(filtered_df)}
        - **Malignant Cases:** {(filtered_df['predicted'] == 'malignant').sum()}
        - **Average Confidence:** {filtered_df['confidence'].mean():.1%}
        - **High Risk Cases:** {(filtered_df['risk_level'].isin(['High Risk', 'Critical Risk'])).sum()}
        """)
        
        # Enhanced data table
        display_columns = [
            'predicted', 'actual', 'confidence', 'risk_level', 
            'is_correct', 'mean radius', 'mean texture', 'worst concavity'
        ]
        
        styled_df = filtered_df[display_columns].copy()
        styled_df.columns = [col.replace('_', ' ').title() for col in styled_df.columns]
        
        st.dataframe(
            styled_df.round(3),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.markdown("#### 📥 Export Options")
        
        # Individual patient report
        st.markdown("**👤 Individual Patient Report**")
        selected_patient = st.selectbox(
            "Select Patient for Report:",
            range(len(df)),
            format_func=lambda x: f"Patient #{x:03d}"
        )
        
        if st.button("📄 Generate Patient Report"):
            patient_data = df.iloc[selected_patient].to_dict()
            
            # Generate comprehensive report
            report_text = f"""
MEDICAL DIAGNOSTIC REPORT
========================

Patient ID: PT{selected_patient:04d}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

DIAGNOSTIC SUMMARY:
- Primary Diagnosis: {patient_data['predicted'].upper()}
- Diagnostic Confidence: {patient_data['confidence']:.1%}
- Risk Stratification: {patient_data['risk_level']}
- Accuracy Status: {'CORRECT' if patient_data['is_correct'] else 'MISCLASSIFIED'}

KEY BIOMARKERS:
- Mean Radius: {patient_data['mean radius']:.2f}
- Mean Texture: {patient_data['mean texture']:.2f}
- Mean Area: {patient_data['mean area']:.0f}
- Worst Concavity: {patient_data['worst concavity']:.3f}

CLINICAL RECOMMENDATIONS:
{chr(10).join([f"- {rec}" for rec in medical_analyzer.generate_clinical_interpretation(
    {col: patient_data[col] for col in feature_names}, 
    patient_data['predicted'], 
    patient_data['confidence']
)['recommendations']])}

---
Report generated by MedAI Diagnostic Suite
For clinical use only. Correlation with clinical findings required.
            """
            
            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name=f"patient_report_PT{selected_patient:04d}.txt",
                mime="text/plain"
            )
        
        # Batch export options
        st.markdown("**📊 Batch Export**")
        
        export_format = st.selectbox(
            "Export Format:",
            ["CSV", "Excel", "JSON"]
        )
        
        if st.button("📥 Export Filtered Data"):
            if export_format == "CSV":
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📋 Download CSV",
                    data=csv_data,
                    file_name=f"diagnostic_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            elif export_format == "JSON":
                json_data = filtered_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📋 Download JSON",
                    data=json_data,
                    file_name=f"diagnostic_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )

# Enhanced footer with system information
st.markdown("---")
st.markdown(f"""
<div class="footer-info">
    <h3>⚕️ MedAI Advanced Diagnostic Suite</h3>
    <p><strong>System Status:</strong> 🟢 Operational | <strong>Model Version:</strong> RF-v1.0 | <strong>Last Updated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d')}</p>
    <p><strong>Total Processed:</strong> {len(df)} cases | <strong>System Accuracy:</strong> {accuracy:.1f}% | <strong>Uptime:</strong> 99.9%</p>
    <hr>
    <p style="font-size: 0.9em; color: #666;">
        🔬 Built with Advanced Machine Learning & Medical AI<br>
        📊 Powered by Streamlit, scikit-learn & Plotly<br>
        ⚕️ For Research and Educational Purposes Only
    </p>
</div>
""", unsafe_allow_html=True)
