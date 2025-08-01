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

# Import custom modules with error handling
try:
    from medical_utils import MedicalAnalyzer, ClinicalReportGenerator, get_medical_name
    MEDICAL_UTILS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Medical utils module not fully available: {e}")
    MEDICAL_UTILS_AVAILABLE = False
    # Fallback function
    def get_medical_name(feature_name):
        medical_names = {
            'mean radius': 'Tumor Radius',
            'mean texture': 'Tissue Texture', 
            'mean perimeter': 'Tumor Perimeter',
            'mean area': 'Tumor Area',
            'mean smoothness': 'Tissue Smoothness',
            'worst concavity': 'Max Concavity',
            'worst symmetry': 'Asymmetry Index'
        }
        return medical_names.get(feature_name, feature_name.title())

try:
    from advanced_visualizations import Medical3DVisualizer, InteractivePlotBuilder, MedicalImageSimulator
    VISUALIZATIONS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Advanced visualizations module not fully available: {e}")
    VISUALIZATIONS_AVAILABLE = False

try:
    from realtime_simulator import create_live_monitoring_page
    REALTIME_AVAILABLE = True
except ImportError as e:
    st.warning(f"Real-time simulator module not fully available: {e}")
    REALTIME_AVAILABLE = False

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

/* CSS Variables for adaptive theming */
:root {
    --card-background: #f8f9fa;
    --text-color: #333;
    --border-color: #e9ecef;
}

/* Dark mode detection and overrides */
@media (prefers-color-scheme: dark) {
    :root {
        --card-background: rgba(40, 42, 54, 0.8);
        --text-color: #f8f9fa;
        --border-color: rgba(255, 255, 255, 0.1);
    }
}

/* Streamlit dark theme detection */
[data-theme="dark"] {
    --card-background: rgba(40, 42, 54, 0.8) !important;
    --text-color: #f8f9fa !important;
    --border-color: rgba(255, 255, 255, 0.1) !important;
}

/* Additional fallback for Streamlit dark backgrounds */
.st-emotion-cache-13k62yr {
    --card-background: rgba(40, 42, 54, 0.8) !important;
    --text-color: #f8f9fa !important;
    --border-color: rgba(255, 255, 255, 0.1) !important;
}

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
    background: var(--card-background, white);
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
    margin: 1rem 0;
    transition: transform 0.2s ease;
    color: var(--text-color, #333);
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
    background: var(--card-background, #f8f9fa);
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border: 1px solid var(--border-color, #e9ecef);
    color: var(--text-color, #333);
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
    background: var(--card-background, white);
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid var(--border-color, #e9ecef);
    text-align: center;
    color: var(--text-color, #333);
}

.navigation-tabs {
    background: var(--card-background, white);
    border-radius: 10px;
    padding: 0.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    color: var(--text-color, #333);
}

.footer-info {
    background: var(--card-background, #f8f9fa);
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    margin-top: 3rem;
    border: 1px solid var(--border-color, #e9ecef);
    color: var(--text-color, #333);
}

/* Force dark mode styles when detected */
body[class*="dark"] .metric-card,
body[class*="dark"] .sidebar-section,
body[class*="dark"] .footer-info,
body[class*="dark"] .feature-item,
body[class*="dark"] .navigation-tabs {
    background: rgba(40, 42, 54, 0.8) !important;
    color: #f8f9fa !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
}
</style>

<script>
// Enhanced dark mode detection
function adaptToTheme() {
    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const streamlitTheme = document.body.style.backgroundColor;
    const hasStreamlitDark = streamlitTheme.includes('14, 17, 23') || streamlitTheme.includes('rgb(14, 17, 23)');
    
    if (isDark || hasStreamlitDark) {
        document.documentElement.style.setProperty('--card-background', 'rgba(40, 42, 54, 0.8)');
        document.documentElement.style.setProperty('--text-color', '#f8f9fa');
        document.documentElement.style.setProperty('--border-color', 'rgba(255, 255, 255, 0.1)');
    } else {
        document.documentElement.style.setProperty('--card-background', '#f8f9fa');
        document.documentElement.style.setProperty('--text-color', '#333');
        document.documentElement.style.setProperty('--border-color', '#e9ecef');
    }
}

// Run immediately and on changes
adaptToTheme();
if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addListener(adaptToTheme);
}

// Monitor for Streamlit theme changes
const observer = new MutationObserver(adaptToTheme);
observer.observe(document.body, { 
    attributes: true, 
    attributeFilter: ['style', 'class'] 
});
</script>
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
if MEDICAL_UTILS_AVAILABLE:
    try:
        medical_analyzer = MedicalAnalyzer()
        report_generator = ClinicalReportGenerator(medical_analyzer)
    except Exception as e:
        st.error(f"Error initializing medical analyzer: {e}")
        medical_analyzer = None
        report_generator = None
        MEDICAL_UTILS_AVAILABLE = False
else:
    medical_analyzer = None
    report_generator = None

if VISUALIZATIONS_AVAILABLE:
    try:
        visualizer_3d = Medical3DVisualizer()
        plot_builder = InteractivePlotBuilder()
        image_simulator = MedicalImageSimulator()
    except Exception as e:
        st.error(f"Error initializing visualizers: {e}")
        visualizer_3d = None
        plot_builder = None
        image_simulator = None
        VISUALIZATIONS_AVAILABLE = False
else:
    visualizer_3d = None
    plot_builder = None
    image_simulator = None

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
    def simple_risk_assessment(confidence, prediction):
        if confidence < 0.7:
            return 'Critical Risk'
        elif confidence < 0.85:
            return 'High Risk' if prediction == 'malignant' else 'Medium Risk'
        else:
            return 'Medium Risk' if prediction == 'malignant' else 'Low Risk'
    
    if MEDICAL_UTILS_AVAILABLE and medical_analyzer:
        df['risk_level'] = df.apply(lambda row: 
            medical_analyzer.assess_risk_level(row['confidence'], row['predicted']).value, axis=1)
    else:
        df['risk_level'] = df.apply(lambda row: 
            simple_risk_assessment(row['confidence'], row['predicted']), axis=1)
    
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
    if VISUALIZATIONS_AVAILABLE and plot_builder:
        try:
            fig_dashboard = plot_builder.create_diagnostic_dashboard_plot(df)
            st.plotly_chart(fig_dashboard, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating advanced dashboard: {e}")
            VISUALIZATIONS_AVAILABLE = False
    
    if not VISUALIZATIONS_AVAILABLE:
        # Create comprehensive fallback analytics dashboard
        st.markdown("#### 🎯 Comprehensive Performance Analytics")
        
        # Create multi-panel analytics dashboard
        fig_analytics = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                '🎯 Model Accuracy', '📊 Confidence Distribution', '⚠️ Risk Stratification',
                '🔬 Prediction Matrix', '📈 Feature Correlations', '🎲 Sample Distribution',
                '💯 Performance Metrics', '🔍 Error Analysis', '📋 Classification Report'
            ),
            specs=[
                [{"type": "pie"}, {"type": "histogram"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}, {"type": "table"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.06
        )
        
        # 1. Model Accuracy Pie Chart
        correct_count = df['is_correct'].sum()
        incorrect_count = len(df) - correct_count
        
        fig_analytics.add_trace(go.Pie(
            values=[correct_count, incorrect_count],
            labels=['Correct', 'Incorrect'],
            hole=0.4,
            marker_colors=['#10b981', '#ef4444'],
            textinfo='label+percent',
            name="Accuracy"
        ), row=1, col=1)
        
        # 2. Confidence Distribution
        fig_analytics.add_trace(go.Histogram(
            x=df['confidence'],
            nbinsx=20,
            marker_color='#3b82f6',
            opacity=0.7,
            name="Confidence"
        ), row=1, col=2)
        
        # 3. Risk Stratification
        risk_counts = df['risk_level'].value_counts()
        risk_colors = {'Critical Risk': '#dc2626', 'High Risk': '#f59e0b', 'Medium Risk': '#3b82f6', 'Low Risk': '#10b981'}
        
        fig_analytics.add_trace(go.Bar(
            x=list(risk_counts.index),
            y=list(risk_counts.values),
            marker_color=[risk_colors.get(level, '#6b7280') for level in risk_counts.index],
            name="Risk Levels"
        ), row=1, col=3)
        
        # 4. Prediction Scatter (PCA space)
        colors_pred = ['#ef4444' if pred == 'malignant' else '#10b981' for pred in df['predicted']]
        fig_analytics.add_trace(go.Scatter(
            x=df['PCA1'],
            y=df['PCA2'],
            mode='markers',
            marker=dict(
                color=colors_pred,
                size=8,
                opacity=0.7
            ),
            text=[f"Patient {i}: {row['predicted']}" for i, row in df.iterrows()],
            name="Predictions"
        ), row=2, col=1)
        
        # 5. Feature Correlations Heatmap
        key_features = ['mean radius', 'mean texture', 'mean area', 'worst concavity', 'worst symmetry']
        corr_matrix = df[key_features].corr()
        
        fig_analytics.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            name="Correlations"
        ), row=2, col=2)
        
        # 6. Sample Distribution (t-SNE space)
        colors_actual = ['#ef4444' if actual == 'malignant' else '#10b981' for actual in df['actual']]
        fig_analytics.add_trace(go.Scatter(
            x=df['TSNE1'],
            y=df['TSNE2'],
            mode='markers',
            marker=dict(
                color=colors_actual,
                size=6,
                opacity=0.6
            ),
            text=[f"Actual: {row['actual']}" for _, row in df.iterrows()],
            name="Actual Labels"
        ), row=2, col=3)
        
        # 7. Performance Metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        y_true_binary = [1 if x == 'malignant' else 0 for x in df['actual']]
        y_pred_binary = [1 if x == 'malignant' else 0 for x in df['predicted']]
        y_prob_malignant = [clf.predict_proba(X[i].reshape(1, -1))[0][0] for i in range(len(X))]
        
        metrics = {
            'Accuracy': accuracy/100,
            'Precision': precision_score(y_true_binary, y_pred_binary),
            'Recall': recall_score(y_true_binary, y_pred_binary),
            'F1-Score': f1_score(y_true_binary, y_pred_binary),
            'ROC-AUC': roc_auc_score(y_true_binary, y_prob_malignant)
        }
        
        fig_analytics.add_trace(go.Bar(
            x=list(metrics.keys()),
            y=[v * 100 for v in metrics.values()],
            marker_color='#8b5cf6',
            name="Metrics (%)"
        ), row=3, col=1)
        
        # 8. Error Analysis
        error_types = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
        tp = sum((df['actual'] == 'malignant') & (df['predicted'] == 'malignant'))
        tn = sum((df['actual'] == 'benign') & (df['predicted'] == 'benign'))
        fp = sum((df['actual'] == 'benign') & (df['predicted'] == 'malignant'))
        fn = sum((df['actual'] == 'malignant') & (df['predicted'] == 'benign'))
        
        error_counts = [tp, tn, fp, fn]
        error_colors = ['#10b981', '#10b981', '#ef4444', '#ef4444']
        
        fig_analytics.add_trace(go.Bar(
            x=error_types,
            y=error_counts,
            marker_color=error_colors,
            name="Confusion Matrix"
        ), row=3, col=2)
        
        # Update layout
        fig_analytics.update_layout(
            height=1200,
            title_text="🎯 Comprehensive Diagnostic Analytics Dashboard",
            showlegend=False
        )
        
        st.plotly_chart(fig_analytics, use_container_width=True)
    
    # Detailed performance metrics in columns
    col1, col2, col3 = st.columns(3)
    
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
            color = "#10b981" if row['Percentage'] > 85 else "#f59e0b" if row['Percentage'] > 70 else "#ef4444"
            st.markdown(f"""
            <div class="metric-card">
                <h4>{row['Metric']}</h4>
                <h2 style="color: {color};">{row['Percentage']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🎯 Feature Importance Analysis")
        
        # Get feature importance from model if available
        if hasattr(clf, 'feature_importances_'):
            feature_importance = clf.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False).head(10)
        else:
            # Fallback: correlation with target
            correlations = []
            for feature in feature_names:
                corr = np.corrcoef(df[feature], y_pred_binary)[0, 1]
                correlations.append(abs(corr))
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': correlations
            }).sort_values('importance', ascending=False).head(10)
        
        fig_importance = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title="🔝 Top 10 Diagnostic Features",
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col3:
        st.markdown("#### 📊 Clinical Insights")
        
        # Clinical statistics
        malignant_cases = (df['predicted'] == 'malignant').sum()
        benign_cases = (df['predicted'] == 'benign').sum()
        high_conf_cases = (df['confidence'] > 0.9).sum()
        low_conf_cases = (df['confidence'] < 0.7).sum()
        
        misclassified = (~df['is_correct']).sum()
        false_positives = ((df['actual'] == 'benign') & (df['predicted'] == 'malignant')).sum()
        false_negatives = ((df['actual'] == 'malignant') & (df['predicted'] == 'benign')).sum()
        
        st.markdown(f"""
        **📋 Diagnostic Summary:**
        - Total Cases Analyzed: {len(df)}
        - Malignant Detected: {malignant_cases} ({malignant_cases/len(df)*100:.1f}%)
        - Benign Detected: {benign_cases} ({benign_cases/len(df)*100:.1f}%)
        
        **🎯 Confidence Analysis:**
        - High Confidence (>90%): {high_conf_cases}
        - Low Confidence (<70%): {low_conf_cases}
        - Average Confidence: {df['confidence'].mean():.1%}
        
        **⚠️ Error Analysis:**
        - Total Misclassifications: {misclassified}
        - False Positives: {false_positives}
        - False Negatives: {false_negatives}
        
        **📈 Model Reliability:**
        - Precision: {precision_score(y_true_binary, y_pred_binary):.1%}
        - Recall: {recall_score(y_true_binary, y_pred_binary):.1%}
        - Overall Accuracy: {accuracy:.1f}%
        """)
        
        # Risk distribution pie chart
        risk_counts = df['risk_level'].value_counts()
        fig_risk_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="🚨 Risk Distribution",
            color_discrete_map={
                'Critical Risk': '#dc2626',
                'High Risk': '#f59e0b', 
                'Medium Risk': '#3b82f6',
                'Low Risk': '#10b981'
            }
        )
        fig_risk_pie.update_layout(height=300)
        st.plotly_chart(fig_risk_pie, use_container_width=True)
    
    # Advanced Analytics Section
    st.markdown("---")
    st.markdown("#### 🔬 Advanced Statistical Analysis")
    
    # Create advanced analytics tabs
    analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
        "📊 Correlation Analysis", 
        "📈 Distribution Analysis", 
        "🎯 Prediction Analysis",
        "📋 Detailed Reports"
    ])
    
    with analysis_tab1:
        st.markdown("##### 🔗 Feature Correlation Matrix")
        
        # Select key features for correlation analysis
        selected_features = st.multiselect(
            "Select features for correlation analysis:",
            feature_names,
            default=['mean radius', 'mean texture', 'mean area', 'worst concavity', 'worst symmetry']
        )
        
        if selected_features:
            corr_matrix = df[selected_features].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="🔗 Feature Correlation Network"
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Correlation insights
            st.markdown("**🔍 Key Correlations:**")
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_val
                        })
            
            if high_corr_pairs:
                corr_df = pd.DataFrame(high_corr_pairs)
                st.dataframe(corr_df.round(3), use_container_width=True)
            else:
                st.info("No strong correlations (>0.7) found among selected features.")
    
    with analysis_tab2:
        st.markdown("##### 📈 Feature Distribution Analysis")
        
        # Feature distribution comparison
        selected_feature = st.selectbox(
            "Select feature for distribution analysis:",
            feature_names,
            index=0
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution by actual diagnosis
            fig_dist_actual = px.histogram(
                df, x=selected_feature, color='actual',
                nbins=20, opacity=0.7,
                title=f"📊 {get_medical_name(selected_feature)} - Actual Diagnosis",
                color_discrete_map={'malignant': '#ef4444', 'benign': '#10b981'}
            )
            st.plotly_chart(fig_dist_actual, use_container_width=True)
        
        with col2:
            # Distribution by predicted diagnosis
            fig_dist_pred = px.histogram(
                df, x=selected_feature, color='predicted',
                nbins=20, opacity=0.7,
                title=f"🎯 {get_medical_name(selected_feature)} - Predicted Diagnosis",
                color_discrete_map={'malignant': '#ef4444', 'benign': '#10b981'}
            )
            st.plotly_chart(fig_dist_pred, use_container_width=True)
        
        # Statistical summary
        st.markdown("**📊 Statistical Summary:**")
        summary_stats = df.groupby('actual')[selected_feature].describe()
        st.dataframe(summary_stats.round(3), use_container_width=True)
    
    with analysis_tab3:
        st.markdown("##### 🎯 Prediction Confidence Analysis")
        
        # Confidence vs Features analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence distribution by diagnosis
            fig_conf_diag = px.box(
                df, x='predicted', y='confidence',
                color='predicted',
                title="📊 Confidence by Prediction",
                color_discrete_map={'malignant': '#ef4444', 'benign': '#10b981'}
            )
            st.plotly_chart(fig_conf_diag, use_container_width=True)
        
        with col2:
            # Accuracy by confidence bins
            df['confidence_bin'] = pd.cut(df['confidence'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            accuracy_by_conf = df.groupby('confidence_bin')['is_correct'].mean() * 100
            
            fig_acc_conf = px.bar(
                x=accuracy_by_conf.index,
                y=accuracy_by_conf.values,
                title="🎯 Accuracy by Confidence Level",
                color=accuracy_by_conf.values,
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_acc_conf, use_container_width=True)
        
        # Misclassification analysis
        st.markdown("**⚠️ Misclassification Analysis:**")
        misclassified_df = df[~df['is_correct']]
        
        if len(misclassified_df) > 0:
            st.markdown(f"Found {len(misclassified_df)} misclassified cases:")
            
            display_cols = ['predicted', 'actual', 'confidence', 'risk_level', 'mean radius', 'mean texture', 'worst concavity']
            st.dataframe(
                misclassified_df[display_cols].round(3),
                use_container_width=True
            )
        else:
            st.success("🎉 Perfect classification! No misclassified cases found.")
    
    with analysis_tab4:
        st.markdown("##### 📋 Comprehensive Analytics Report")
        
        # Generate comprehensive report
        report_data = {
            "Dataset Overview": {
                "Total Samples": len(df),
                "Features": len(feature_names),
                "Malignant Cases": (df['actual'] == 'malignant').sum(),
                "Benign Cases": (df['actual'] == 'benign').sum(),
            },
            "Model Performance": {
                "Overall Accuracy": f"{accuracy:.2f}%",
                "Precision": f"{precision_score(y_true_binary, y_pred_binary):.3f}",
                "Recall": f"{recall_score(y_true_binary, y_pred_binary):.3f}",
                "F1-Score": f"{f1_score(y_true_binary, y_pred_binary):.3f}",
                "ROC-AUC": f"{roc_auc_score(y_true_binary, y_prob_malignant):.3f}",
            },
            "Risk Analysis": {
                "High Risk Cases": (df['risk_level'].isin(['High Risk', 'Critical Risk'])).sum(),
                "Average Confidence": f"{df['confidence'].mean():.1%}",
                "Low Confidence Cases (<70%)": (df['confidence'] < 0.7).sum(),
            },
            "Error Analysis": {
                "Total Errors": misclassified,
                "False Positives": false_positives,
                "False Negatives": false_negatives,
                "Error Rate": f"{(misclassified/len(df)*100):.2f}%",
            }
        }
        
        # Display report in organized format
        for section, data in report_data.items():
            st.markdown(f"**{section}:**")
            for key, value in data.items():
                st.write(f"- {key}: {value}")
            st.markdown("---")
        
        # Export full analytics report
        if st.button("📥 Export Full Analytics Report"):
            # Create comprehensive CSV report
            analytics_export = df.copy()
            analytics_export['confidence_bin'] = pd.cut(analytics_export['confidence'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            csv_data = analytics_export.to_csv(index=False)
            st.download_button(
                label="📊 Download Analytics CSV",
                data=csv_data,
                file_name=f"full_analytics_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
            # Create summary report
            summary_report = f"""
COMPREHENSIVE ANALYTICS REPORT
============================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW:
- Total Samples: {len(df)}
- Features: {len(feature_names)}
- Malignant Cases: {(df['actual'] == 'malignant').sum()}
- Benign Cases: {(df['actual'] == 'benign').sum()}

MODEL PERFORMANCE:
- Overall Accuracy: {accuracy:.2f}%
- Precision: {precision_score(y_true_binary, y_pred_binary):.3f}
- Recall: {recall_score(y_true_binary, y_pred_binary):.3f}
- F1-Score: {f1_score(y_true_binary, y_pred_binary):.3f}
- ROC-AUC: {roc_auc_score(y_true_binary, y_prob_malignant):.3f}

RISK ANALYSIS:
- High Risk Cases: {(df['risk_level'].isin(['High Risk', 'Critical Risk'])).sum()}
- Average Confidence: {df['confidence'].mean():.1%}
- Low Confidence Cases (<70%): {(df['confidence'] < 0.7).sum()}

ERROR ANALYSIS:
- Total Errors: {misclassified}
- False Positives: {false_positives}
- False Negatives: {false_negatives}
- Error Rate: {(misclassified/len(df)*100):.2f}%

TOP FEATURES (by importance):
{chr(10).join([f"- {row['feature']}: {row['importance']:.3f}" for _, row in importance_df.head(5).iterrows()])}

---
Report generated by MedAI Enhanced Diagnostic Suite
For research and educational purposes only.
            """
            
            st.download_button(
                label="📋 Download Summary Report",
                data=summary_report,
                file_name=f"analytics_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

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
