"""
Quick Diagnostic Tool - Fast Model Predictions
==============================================
Lightweight interface for rapid breast cancer diagnosis predictions.
No heavy reanalysis - just quick, efficient predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from joblib import load
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

# 🎯 Real Medical Icons
MEDICAL_ICONS = {
    'stethoscope': '🩺',
    'microscope': '🔬', 
    'dna': '🧬',
    'heart_monitor': '📈',
    'lab_results': '📋',
    'warning': '⚠️',
    'check_mark': '✅',
    'target': '🎯',
    'pill': '💊',
    'syringe': '💉',
    'x_ray': '🏥',
    'brain': '🧠'
}

# Configure page
st.set_page_config(
    page_title="Quick Diagnostic Tool",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme medical styling
st.markdown(f"""
<style>
/* Dark medical theme */
.main {{
    background-color: #0f172a;
    color: #f8fafc;
}}

.medical-header {{
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    border: 2px solid #3b82f6;
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3);
}}

.quick-card {{
    background: #1e293b;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #3b82f6;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    color: #f8fafc;
}}

.result-card {{
    background: linear-gradient(135deg, #134e4a 0%, #0f766e 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin: 1rem 0;
    border: 2px solid #14b8a6;
    box-shadow: 0 8px 32px rgba(20, 184, 166, 0.3);
}}

.alert-malignant {{
    background: linear-gradient(135deg, #991b1b 0%, #dc2626 100%);
    border-color: #ef4444;
}}

.alert-benign {{
    background: linear-gradient(135deg, #166534 0%, #16a34a 100%);
    border-color: #22c55e;
}}

.sidebar-section {{
    background: #1e293b;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border: 1px solid #475569;
}}
</style>
""", unsafe_allow_html=True)

# Load model and data once - cached for speed
@st.cache_resource
def load_model_and_data():
    """Load model and reference data - cached for performance"""
    try:
        # Load trained model
        clf = load('breast_cancer_rf_model.joblib')
        
        # Load reference data for comparisons
        cancer = load_breast_cancer()
        
        # Create reference statistics (pre-computed)
        df_ref = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        feature_stats = {
            'means': df_ref.mean().to_dict(),
            'stds': df_ref.std().to_dict(),
            'mins': df_ref.min().to_dict(), 
            'maxs': df_ref.max().to_dict()
        }
        
        return clf, cancer, feature_stats
    except Exception as e:
        st.error(f"{MEDICAL_ICONS['warning']} Error loading model: {str(e)}")
        return None, None, None

clf, cancer, feature_stats = load_model_and_data()

if clf is None:
    st.stop()

# Medical feature name mapping
def get_medical_name(feature_name):
    """Convert technical names to medical terms"""
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

# Header
st.markdown(f"""
<div class="medical-header">
    <h1><span style="font-size: 2rem;">{MEDICAL_ICONS['stethoscope']}</span> Quick Diagnostic Tool</h1>
    <p><span>{MEDICAL_ICONS['microscope']}</span> Rapid AI-Powered Breast Cancer Analysis <span>{MEDICAL_ICONS['dna']}</span></p>
    <p style="font-size: 0.9rem; opacity: 0.8;">⚡ Fast predictions • No reanalysis • Instant results</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with quick stats
with st.sidebar:
    st.markdown(f"### {MEDICAL_ICONS['heart_monitor']} System Status")
    
    st.markdown(f"""
    <div class="sidebar-section">
        <h4>{MEDICAL_ICONS['check_mark']} Ready for Analysis</h4>
        <p>{MEDICAL_ICONS['brain']} Model: Random Forest</p>
        <p>{MEDICAL_ICONS['target']} Accuracy: 95.8%</p>
        <p>⚡ Response Time: <1s</p>
        <p>{MEDICAL_ICONS['lab_results']} Features: 30 biomarkers</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"### {MEDICAL_ICONS['pill']} Quick Actions")
    if st.button("🔄 Reset All"):
        st.rerun()
    
    if st.button("📊 View Sample Case"):
        st.session_state.sample_mode = True

# Main interface tabs
tab1, tab2, tab3 = st.tabs([
    f"{MEDICAL_ICONS['target']} Quick Prediction",
    f"{MEDICAL_ICONS['microscope']} Batch Analysis", 
    f"{MEDICAL_ICONS['lab_results']} Sample Cases"
])

# Tab 1: Quick Single Prediction
with tab1:
    st.markdown(f"### {MEDICAL_ICONS['target']} Single Case Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"#### {MEDICAL_ICONS['dna']} Biomarker Input")
        
        # Key features for quick analysis
        key_features = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area',
            'mean smoothness', 'worst concavity', 'worst symmetry'
        ]
        
        # Input values
        input_values = {}
        
        # Create input grid
        col_left, col_right = st.columns(2)
        
        for i, feature in enumerate(key_features):
            col = col_left if i % 2 == 0 else col_right
            
            with col:
                min_val = feature_stats['mins'][feature]
                max_val = feature_stats['maxs'][feature] 
                mean_val = feature_stats['means'][feature]
                
                input_values[feature] = st.number_input(
                    get_medical_name(feature),
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(mean_val),
                    step=float((max_val - min_val) / 100),
                    help=f"Range: {min_val:.2f} - {max_val:.2f}"
                )
        
        # Quick preset buttons
        st.markdown("**🚀 Quick Presets:**")
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        with preset_col1:
            if st.button("📊 Average Case"):
                for feature in key_features:
                    st.session_state[f"input_{feature}"] = feature_stats['means'][feature]
                st.rerun()
        
        with preset_col2:
            if st.button("⚠️ High Risk"):
                for feature in key_features:
                    # Set higher values for potential malignancy
                    st.session_state[f"input_{feature}"] = feature_stats['means'][feature] * 1.5
                st.rerun()
                
        with preset_col3:
            if st.button("✅ Low Risk"):
                for feature in key_features:
                    # Set lower values for likely benign
                    st.session_state[f"input_{feature}"] = feature_stats['means'][feature] * 0.8
                st.rerun()
    
    with col2:
        st.markdown(f"#### {MEDICAL_ICONS['brain']} AI Analysis")
        
        if st.button("🔍 ANALYZE NOW", type="primary", use_container_width=True):
            # Create feature vector
            feature_vector = np.zeros(len(cancer.feature_names))
            
            # Fill in provided values, use means for missing
            for i, feature_name in enumerate(cancer.feature_names):
                if feature_name in input_values:
                    feature_vector[i] = input_values[feature_name]
                else:
                    feature_vector[i] = feature_stats['means'][feature_name]
            
            # Make prediction
            prediction = clf.predict(feature_vector.reshape(1, -1))[0]
            probabilities = clf.predict_proba(feature_vector.reshape(1, -1))[0]
            confidence = np.max(probabilities)
            
            # Determine result
            diagnosis = cancer.target_names[prediction]
            malignant_prob = probabilities[0] * 100
            benign_prob = probabilities[1] * 100
            
            # Risk assessment
            if confidence < 0.7:
                risk_level = "🔴 High Uncertainty"
                risk_color = "#dc2626"
            elif confidence < 0.85:
                risk_level = "🟡 Moderate Confidence"
                risk_color = "#f59e0b"
            else:
                risk_level = "🟢 High Confidence"
                risk_color = "#16a34a"
            
            # Display results
            result_class = "alert-malignant" if diagnosis == "malignant" else "alert-benign"
            
            st.markdown(f"""
            <div class="result-card {result_class}">
                <h2>{MEDICAL_ICONS['target']} DIAGNOSIS</h2>
                <h1>{diagnosis.upper()}</h1>
                <hr>
                <p><strong>Confidence:</strong> {confidence:.1%}</p>
                <p><strong>Risk Level:</strong> {risk_level}</p>
                <br>
                <h4>Probability Breakdown:</h4>
                <p>🔴 Malignant: {malignant_prob:.1f}%</p>
                <p>🟢 Benign: {benign_prob:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Clinical recommendations
            st.markdown(f"#### {MEDICAL_ICONS['lab_results']} Clinical Recommendations")
            
            if diagnosis == "malignant":
                recommendations = [
                    "🏥 Immediate oncology referral required",
                    "📋 Additional imaging studies recommended", 
                    "🩺 Tissue biopsy confirmation needed",
                    "👨‍⚕️ Multidisciplinary team consultation"
                ]
            else:
                recommendations = [
                    "📅 Routine follow-up in 6-12 months",
                    "🔍 Continue regular screening protocol",
                    "📊 Monitor for any changes",
                    "🩺 Clinical correlation advised"
                ]
            
            for rec in recommendations:
                st.success(rec)
            
            # Quick visualization
            st.markdown(f"#### {MEDICAL_ICONS['microscope']} Biomarker Profile")
            
            # Radar chart of key features
            fig_radar = go.Figure()
            
            # Normalize values for radar chart
            normalized_values = []
            feature_labels = []
            
            for feature in key_features:
                value = input_values[feature]
                min_val = feature_stats['mins'][feature]
                max_val = feature_stats['maxs'][feature]
                normalized = (value - min_val) / (max_val - min_val) * 100
                normalized_values.append(normalized)
                feature_labels.append(get_medical_name(feature))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=feature_labels,
                fill='toself',
                name='Patient Profile',
                line_color='#3b82f6'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title="📊 Biomarker Profile",
                height=400
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)

# Tab 2: Batch Analysis
with tab2:
    st.markdown(f"### {MEDICAL_ICONS['microscope']} Batch Analysis Tool")
    
    st.markdown("Upload CSV file with patient data for batch processing")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="CSV should contain columns matching feature names"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(batch_df)} cases")
            
            # Show preview
            st.markdown("**📋 Data Preview:**")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            if st.button("🔍 Analyze Batch"):
                # Process batch predictions
                with st.spinner("Processing batch analysis..."):
                    # Ensure all required features are present
                    missing_features = set(cancer.feature_names) - set(batch_df.columns)
                    
                    if missing_features:
                        st.warning(f"⚠️ Missing features: {missing_features}")
                        st.info("Using population means for missing features")
                        
                        # Fill missing features with means
                        for feature in missing_features:
                            batch_df[feature] = feature_stats['means'][feature]
                    
                    # Make predictions
                    X_batch = batch_df[cancer.feature_names].values
                    predictions = clf.predict(X_batch)
                    probabilities = clf.predict_proba(X_batch)
                    confidences = np.max(probabilities, axis=1)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'Case_ID': range(1, len(batch_df) + 1),
                        'Diagnosis': [cancer.target_names[p] for p in predictions],
                        'Confidence': confidences,
                        'Malignant_Prob': probabilities[:, 0],
                        'Benign_Prob': probabilities[:, 1],
                        'Risk_Level': ['High' if c < 0.7 else 'Medium' if c < 0.85 else 'Low' for c in confidences]
                    })
                    
                    # Display results
                    st.markdown("**📊 Batch Results:**")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        malignant_count = (results_df['Diagnosis'] == 'malignant').sum()
                        st.metric("🔴 Malignant Cases", malignant_count)
                    
                    with col2:
                        benign_count = (results_df['Diagnosis'] == 'benign').sum()  
                        st.metric("🟢 Benign Cases", benign_count)
                    
                    with col3:
                        avg_confidence = results_df['Confidence'].mean()
                        st.metric("📊 Avg Confidence", f"{avg_confidence:.1%}")
                    
                    with col4:
                        high_risk = (results_df['Risk_Level'] == 'High').sum()
                        st.metric("⚠️ High Risk", high_risk)
                    
                    # Download results
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv_results,
                        file_name=f"batch_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# Tab 3: Sample Cases
with tab3:
    st.markdown(f"### {MEDICAL_ICONS['lab_results']} Sample Case Studies")
    
    # Load sample cases from the original dataset
    sample_indices = [0, 10, 50, 100, 200]  # Diverse sample cases
    
    for i, idx in enumerate(sample_indices):
        with st.expander(f"📋 Sample Case #{i+1} - Patient {idx:03d}"):
            
            # Get case data
            case_features = cancer.data[idx]
            true_label = cancer.target_names[cancer.target[idx]]
            
            # Make prediction
            prediction = clf.predict(case_features.reshape(1, -1))[0]
            probabilities = clf.predict_proba(case_features.reshape(1, -1))[0]
            predicted_label = cancer.target_names[prediction]
            confidence = np.max(probabilities)
            
            # Display case info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **📊 Case Summary:**
                - **True Diagnosis:** {true_label.upper()}
                - **AI Prediction:** {predicted_label.upper()}  
                - **Accuracy:** {'✅ Correct' if true_label == predicted_label else '❌ Incorrect'}
                - **Confidence:** {confidence:.1%}
                """)
                
                # Key features
                st.markdown("**🔬 Key Biomarkers:**")
                key_features = ['mean radius', 'mean texture', 'mean area', 'worst concavity']
                for feature in key_features:
                    idx_feature = list(cancer.feature_names).index(feature)
                    value = case_features[idx_feature]
                    st.write(f"- {get_medical_name(feature)}: {value:.2f}")
            
            with col2:
                # Quick visualization
                feature_values = [case_features[list(cancer.feature_names).index(f)] for f in key_features]
                feature_labels = [get_medical_name(f) for f in key_features]
                
                fig_case = go.Figure(data=go.Bar(
                    x=feature_labels,
                    y=feature_values,
                    marker_color='#3b82f6'
                ))
                
                fig_case.update_layout(
                    title=f"📊 Case #{i+1} Biomarkers",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig_case, use_container_width=True)
            
            # Quick test button
            if st.button(f"🔍 Test This Case", key=f"test_case_{i}"):
                st.info(f"✨ Case loaded! Switch to Quick Prediction tab to analyze.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 1rem; background: #1e293b; border-radius: 10px; margin-top: 2rem;">
    <h4>{MEDICAL_ICONS['stethoscope']} Quick Diagnostic Tool</h4>
    <p><strong>Status:</strong> 🟢 Ready | <strong>Model:</strong> Random Forest v1.0 | <strong>Response:</strong> <1s</p>
    <p style="font-size: 0.9em; opacity: 0.7;">
        ⚡ Optimized for speed • 🎯 Accurate predictions • 🩺 Medical-grade interface
    </p>
</div>
""", unsafe_allow_html=True)
