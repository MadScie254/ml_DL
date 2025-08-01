"""
Real-time Diagnostic Simulator
==============================
Simulates real-time patient intake and diagnostic workflow
for breast cancer screening and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import threading
import queue
from medical_utils import MedicalAnalyzer, ClinicalReportGenerator
from advanced_visualizations import Medical3DVisualizer

class PatientIntakeSimulator:
    """Simulate real-time patient intake for diagnostic screening"""
    
    def __init__(self, model, feature_names, target_names):
        self.model = model
        self.feature_names = feature_names
        self.target_names = target_names
        self.patient_queue = queue.Queue()
        self.processed_patients = []
        self.is_running = False
        
    def generate_synthetic_patient(self, patient_id: int) -> Dict:
        """Generate realistic synthetic patient data"""
        
        # Base demographics
        age = random.randint(25, 80)
        
        # Generate features with some clinical correlation
        base_radius = random.uniform(8, 25)
        base_texture = random.uniform(8, 35)
        
        # Create correlated features (simplified medical relationships)
        synthetic_features = {
            'mean radius': base_radius + random.normal(0, 1),
            'mean texture': base_texture + random.normal(0, 2),
            'mean perimeter': base_radius * 6.3 + random.normal(0, 5),  # Circumference relation
            'mean area': np.pi * (base_radius ** 2) + random.normal(0, 50),  # Area relation
            'mean smoothness': random.uniform(0.05, 0.15),
            'mean compactness': random.uniform(0.01, 0.3),
            'mean concavity': random.uniform(0, 0.4),
            'mean concave points': random.uniform(0, 0.2),
            'mean symmetry': random.uniform(0.1, 0.3),
            'mean fractal dimension': random.uniform(0.05, 0.1),
            'radius error': random.uniform(0.1, 2.5),
            'texture error': random.uniform(0.3, 4.0),
            'perimeter error': random.uniform(0.7, 20),
            'area error': random.uniform(5, 150),
            'smoothness error': random.uniform(0.001, 0.03),
            'compactness error': random.uniform(0.002, 0.13),
            'concavity error': random.uniform(0, 0.4),
            'concave points error': random.uniform(0, 0.05),
            'symmetry error': random.uniform(0.007, 0.08),
            'fractal dimension error': random.uniform(0.0008, 0.03),
            'worst radius': base_radius * 1.3 + random.normal(0, 2),
            'worst texture': base_texture * 1.2 + random.normal(0, 3),
            'worst perimeter': base_radius * 8 + random.normal(0, 10),
            'worst area': np.pi * ((base_radius * 1.3) ** 2) + random.normal(0, 100),
            'worst smoothness': random.uniform(0.07, 0.22),
            'worst compactness': random.uniform(0.02, 1.0),
            'worst concavity': random.uniform(0, 1.2),
            'worst concave points': random.uniform(0, 0.3),
            'worst symmetry': random.uniform(0.15, 0.7),
            'worst fractal dimension': random.uniform(0.055, 0.2)
        }
        
        # Ensure all 30 features are present
        feature_vector = np.array([synthetic_features.get(name, random.uniform(0, 1)) 
                                 for name in self.feature_names])
        
        # Generate prediction
        prediction = self.model.predict(feature_vector.reshape(1, -1))[0]
        probability = self.model.predict_proba(feature_vector.reshape(1, -1))[0]
        confidence = np.max(probability)
        
        patient_data = {
            'patient_id': f'PT{patient_id:04d}',
            'timestamp': datetime.now(),
            'age': age,
            'features': synthetic_features,
            'feature_vector': feature_vector,
            'prediction': self.target_names[prediction],
            'probability': probability,
            'confidence': confidence,
            'status': 'pending_review' if confidence < 0.8 else 'processed'
        }
        
        return patient_data
    
    def start_simulation(self, interval_seconds: float = 5.0):
        """Start the real-time patient simulation"""
        self.is_running = True
        patient_counter = 1
        
        def simulation_worker():
            while self.is_running:
                patient = self.generate_synthetic_patient(patient_counter)
                self.patient_queue.put(patient)
                time.sleep(interval_seconds)
                patient_counter += 1
        
        simulation_thread = threading.Thread(target=simulation_worker, daemon=True)
        simulation_thread.start()
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
    
    def get_pending_patients(self) -> List[Dict]:
        """Get all patients currently in the queue"""
        patients = []
        while not self.patient_queue.empty():
            try:
                patient = self.patient_queue.get_nowait()
                patients.append(patient)
                self.processed_patients.append(patient)
            except queue.Empty:
                break
        return patients

class RealTimeDashboard:
    """Real-time diagnostic dashboard with live updates"""
    
    def __init__(self):
        self.medical_analyzer = MedicalAnalyzer()
        self.visualizer = Medical3DVisualizer()
    
    def create_live_metrics_display(self, patients_data: List[Dict]) -> Dict[str, any]:
        """Create live metrics for real-time display"""
        
        if not patients_data:
            return {
                'total_patients': 0,
                'malignant_cases': 0,
                'pending_review': 0,
                'average_confidence': 0,
                'hourly_rate': 0
            }
        
        total_patients = len(patients_data)
        malignant_cases = sum(1 for p in patients_data if p['prediction'] == 'malignant')
        pending_review = sum(1 for p in patients_data if p['status'] == 'pending_review')
        avg_confidence = np.mean([p['confidence'] for p in patients_data])
        
        # Calculate hourly processing rate
        if total_patients > 1:
            time_span = (patients_data[-1]['timestamp'] - patients_data[0]['timestamp']).total_seconds() / 3600
            hourly_rate = total_patients / max(time_span, 0.1)
        else:
            hourly_rate = 0
        
        return {
            'total_patients': total_patients,
            'malignant_cases': malignant_cases,
            'pending_review': pending_review,
            'average_confidence': avg_confidence,
            'hourly_rate': hourly_rate,
            'malignancy_rate': (malignant_cases / total_patients * 100) if total_patients > 0 else 0
        }
    
    def create_live_timeline_chart(self, patients_data: List[Dict]) -> go.Figure:
        """Create real-time timeline of patient processing"""
        
        if not patients_data:
            return go.Figure().add_annotation(text="No data available", showarrow=False)
        
        df = pd.DataFrame([{
            'timestamp': p['timestamp'],
            'patient_id': p['patient_id'],
            'prediction': p['prediction'],
            'confidence': p['confidence'],
            'status': p['status']
        } for p in patients_data])
        
        # Create timeline scatter plot
        fig = go.Figure()
        
        colors = {'malignant': '#F44336', 'benign': '#4CAF50'}
        
        for prediction in df['prediction'].unique():
            mask = df['prediction'] == prediction
            subset = df[mask]
            
            fig.add_trace(go.Scatter(
                x=subset['timestamp'],
                y=subset['confidence'],
                mode='markers+lines',
                name=prediction.title(),
                marker=dict(
                    color=colors.get(prediction, '#999'),
                    size=10,
                    symbol='circle' if prediction == 'benign' else 'diamond'
                ),
                hovertemplate=(
                    f"<b>{prediction.title()}</b><br>" +
                    "Patient: %{customdata}<br>" +
                    "Confidence: %{y:.1%}<br>" +
                    "Time: %{x}<br>" +
                    "<extra></extra>"
                ),
                customdata=subset['patient_id']
            ))
        
        fig.update_layout(
            title='⏰ Real-Time Patient Processing Timeline',
            xaxis_title='Time',
            yaxis_title='Diagnostic Confidence',
            yaxis=dict(tickformat='.0%'),
            height=400,
            hovermode='closest'
        )
        
        return fig
    
    def create_patient_flow_diagram(self, patients_data: List[Dict]) -> go.Figure:
        """Create patient flow diagram"""
        
        if not patients_data:
            return go.Figure()
        
        # Calculate flow statistics
        total = len(patients_data)
        benign = sum(1 for p in patients_data if p['prediction'] == 'benign')
        malignant = sum(1 for p in patients_data if p['prediction'] == 'malignant')
        high_conf = sum(1 for p in patients_data if p['confidence'] > 0.9)
        pending = sum(1 for p in patients_data if p['status'] == 'pending_review')
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["Total Patients", "Benign", "Malignant", "High Confidence", "Pending Review", "Cleared"],
                color=["lightblue", "#4CAF50", "#F44336", "#2196F3", "#FF9800", "#9C27B0"]
            ),
            link=dict(
                source=[0, 0, 1, 2, 1, 2, 3],
                target=[1, 2, 3, 3, 4, 4, 5],
                value=[benign, malignant, benign*0.8, malignant*0.7, benign*0.2, malignant*0.3, high_conf]
            )
        )])
        
        fig.update_layout(
            title_text="🔄 Patient Flow Analysis",
            font_size=12,
            height=400
        )
        
        return fig

class AlertSystem:
    """Medical alert and notification system"""
    
    def __init__(self):
        self.alert_thresholds = {
            'high_malignancy_rate': 0.3,  # 30% or higher
            'low_confidence_case': 0.6,   # Below 60% confidence
            'critical_case': 0.95,        # High confidence malignant
            'system_overload': 50         # Too many pending cases
        }
        self.active_alerts = []
    
    def check_alerts(self, patients_data: List[Dict], metrics: Dict) -> List[Dict]:
        """Check for medical alerts and system notifications"""
        
        alerts = []
        current_time = datetime.now()
        
        # Check malignancy rate
        if metrics['malignancy_rate'] > self.alert_thresholds['high_malignancy_rate'] * 100:
            alerts.append({
                'type': 'warning',
                'title': '⚠️ High Malignancy Rate',
                'message': f"Malignancy rate at {metrics['malignancy_rate']:.1f}% - above normal threshold",
                'timestamp': current_time,
                'priority': 'medium'
            })
        
        # Check for low confidence cases
        low_conf_cases = [p for p in patients_data if p['confidence'] < self.alert_thresholds['low_confidence_case']]
        if low_conf_cases:
            alerts.append({
                'type': 'info',
                'title': '🔍 Low Confidence Cases',
                'message': f"{len(low_conf_cases)} cases require manual review",
                'timestamp': current_time,
                'priority': 'low',
                'cases': [p['patient_id'] for p in low_conf_cases]
            })
        
        # Check for critical cases
        critical_cases = [p for p in patients_data 
                         if p['prediction'] == 'malignant' and p['confidence'] > self.alert_thresholds['critical_case']]
        if critical_cases:
            for case in critical_cases:
                alerts.append({
                    'type': 'error',
                    'title': '🚨 Critical Case Alert',
                    'message': f"Patient {case['patient_id']} - High confidence malignant case detected",
                    'timestamp': current_time,
                    'priority': 'high',
                    'patient_id': case['patient_id']
                })
        
        # Check system load
        if metrics['pending_review'] > self.alert_thresholds['system_overload']:
            alerts.append({
                'type': 'warning',
                'title': '📊 System Overload',
                'message': f"{metrics['pending_review']} cases pending review - consider additional resources",
                'timestamp': current_time,
                'priority': 'medium'
            })
        
        return alerts
    
    def format_alert_display(self, alerts: List[Dict]) -> str:
        """Format alerts for display in Streamlit"""
        
        if not alerts:
            return "✅ No active alerts - System operating normally"
        
        alert_html = "<div style='margin: 10px 0;'>"
        
        for alert in sorted(alerts, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True):
            
            color_map = {
                'error': '#ffebee',
                'warning': '#fff3e0', 
                'info': '#e3f2fd'
            }
            
            border_map = {
                'error': '#f44336',
                'warning': '#ff9800',
                'info': '#2196f3'
            }
            
            alert_html += f"""
            <div style='background: {color_map[alert["type"]]}; 
                        border-left: 4px solid {border_map[alert["type"]]}; 
                        padding: 10px; margin: 5px 0; border-radius: 5px;'>
                <strong>{alert['title']}</strong><br>
                {alert['message']}<br>
                <small style='color: #666;'>
                    {alert['timestamp'].strftime('%H:%M:%S')} | Priority: {alert['priority'].upper()}
                </small>
            </div>
            """
        
        alert_html += "</div>"
        return alert_html

def create_live_monitoring_page():
    """Create the live monitoring page layout"""
    
    st.markdown("### 🔴 LIVE Medical Monitoring Center")
    st.markdown("Real-time breast cancer diagnostic screening simulation")
    
    # Initialize session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = None
        st.session_state.dashboard = RealTimeDashboard()
        st.session_state.alert_system = AlertSystem()
        st.session_state.patients_data = []
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Start Simulation", type="primary"):
            # This would need the actual model loaded
            st.info("Simulation would start here with loaded model")
    
    with col2:
        if st.button("⏹️ Stop Simulation"):
            st.info("Simulation stopped")
    
    with col3:
        interval = st.selectbox("Update Interval", [1, 5, 10, 30], index=1)
    
    # Placeholder for live data - in real implementation this would update automatically
    st.markdown("---")
    st.markdown("### 📊 Live Metrics")
    
    # Mock metrics for demonstration
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", "127", "5")
    
    with col2:
        st.metric("Malignant Cases", "23", "2")
    
    with col3:
        st.metric("Pending Review", "8", "-1")
    
    with col4:
        st.metric("Avg Confidence", "87.2%", "1.3%")
    
    return st.session_state.dashboard
