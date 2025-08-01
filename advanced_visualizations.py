"""
Advanced Visualization Module
============================
Immersive 3D visualizations and interactive medical plots
for breast cancer diagnostic analysis.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

class Medical3DVisualizer:
    """Advanced 3D visualization for medical data"""
    
    def __init__(self):
        self.color_scheme = {
            'benign': '#4CAF50',
            'malignant': '#F44336',
            'uncertain': '#FF9800',
            'background': '#f8f9fa'
        }
    
    def create_3d_molecular_space(self, df: pd.DataFrame) -> go.Figure:
        """Create immersive 3D molecular visualization"""
        
        # Use PCA to create 3D coordinates
        pca_3d = PCA(n_components=3)
        coords_3d = pca_3d.fit_transform(df.select_dtypes(include=[np.number]).iloc[:, :-5])
        
        fig = go.Figure()
        
        # Create separate traces for each diagnosis
        for diagnosis in df['predicted'].unique():
            mask = df['predicted'] == diagnosis
            
            fig.add_trace(go.Scatter3d(
                x=coords_3d[mask, 0],
                y=coords_3d[mask, 1], 
                z=coords_3d[mask, 2],
                mode='markers',
                name=diagnosis.title(),
                marker=dict(
                    size=df.loc[mask, 'confidence'] * 15,
                    color=self.color_scheme[diagnosis],
                    opacity=0.8,
                    line=dict(width=2, color='white')
                ),
                hovertemplate=(
                    f"<b>{diagnosis.title()}</b><br>" +
                    "Confidence: %{marker.size:.1%}<br>" +
                    "PC1: %{x:.2f}<br>" +
                    "PC2: %{y:.2f}<br>" +
                    "PC3: %{z:.2f}<br>" +
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title={
                'text': '🧬 3D Molecular Diagnostic Space',
                'x': 0.5,
                'font': {'size': 20}
            },
            scene=dict(
                xaxis_title='Principal Component 1',
                yaxis_title='Principal Component 2',
                zaxis_title='Principal Component 3',
                bgcolor=self.color_scheme['background'],
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=700,
            showlegend=True
        )
        
        return fig
    
    def create_confidence_landscape(self, df: pd.DataFrame) -> go.Figure:
        """Create 3D confidence landscape visualization"""
        
        # Create mesh grid for surface plot
        x = df['PCA1']
        y = df['PCA2'] 
        z = df['confidence']
        
        # Create triangular mesh
        fig = go.Figure(data=[go.Mesh3d(
            x=x,
            y=y,
            z=z,
            colorscale='Viridis',
            intensity=z,
            name='Confidence Surface',
            showscale=True,
            colorbar=dict(title="Confidence Level")
        )])
        
        # Add scatter points for actual predictions
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=4,
                color=df['confidence'],
                colorscale='Plasma',
                opacity=0.8
            ),
            name='Data Points',
            hovertemplate=(
                "Confidence: %{z:.1%}<br>" +
                "Prediction: %{customdata}<br>" +
                "<extra></extra>"
            ),
            customdata=df['predicted']
        ))
        
        fig.update_layout(
            title='🏔️ Diagnostic Confidence Landscape',
            scene=dict(
                xaxis_title='Principal Component 1',
                yaxis_title='Principal Component 2',
                zaxis_title='Confidence Level',
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            height=700
        )
        
        return fig
    
    def create_biomarker_radar_3d(self, patient_data: Dict[str, float]) -> go.Figure:
        """Create 3D radar chart for biomarker analysis"""
        
        features = ['mean radius', 'mean texture', 'mean perimeter', 
                   'mean area', 'worst concavity', 'worst symmetry']
        
        values = [patient_data.get(f, 0) for f in features]
        
        # Normalize values for radar chart
        normalized_values = [(v - min(values)) / (max(values) - min(values)) for v in values]
        
        # Create circular coordinates
        angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        normalized_values += normalized_values[:1]
        
        # Convert to 3D coordinates
        x = [r * np.cos(a) for r, a in zip(normalized_values, angles)]
        y = [r * np.sin(a) for r, a in zip(normalized_values, angles)]
        z = [0.1] * len(x)  # Slight elevation for 3D effect
        
        fig = go.Figure()
        
        # Add the radar surface
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            line=dict(color='#1f77b4', width=8),
            marker=dict(size=8, color='#1f77b4'),
            name='Patient Profile'
        ))
        
        # Add feature labels
        for i, (feature, angle, value) in enumerate(zip(features, angles[:-1], normalized_values[:-1])):
            x_label = 1.2 * np.cos(angle)
            y_label = 1.2 * np.sin(angle)
            
            fig.add_trace(go.Scatter3d(
                x=[x_label], y=[y_label], z=[0.2],
                mode='text',
                text=[feature.replace('_', ' ').title()],
                textposition='middle center',
                showlegend=False,
                textfont=dict(size=12, color='black')
            ))
        
        fig.update_layout(
            title='🎯 3D Biomarker Profile Analysis',
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False), 
                zaxis=dict(visible=False),
                camera=dict(eye=dict(x=0, y=0, z=2))
            ),
            height=600,
            showlegend=False
        )
        
        return fig

class InteractivePlotBuilder:
    """Build complex interactive medical visualizations"""
    
    def __init__(self):
        pass
    
    def create_diagnostic_dashboard_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create comprehensive diagnostic dashboard"""
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Confidence Distribution', 'Risk Level Analysis', 'PCA Clustering',
                'Feature Correlation', 'Prediction Accuracy', 't-SNE Embedding',
                'Biomarker Boxplots', 'ROC Analysis', 'Confusion Heatmap'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "pie"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "box"}, {"type": "scatter"}, {"type": "heatmap"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.06
        )
        
        # 1. Confidence distribution
        fig.add_trace(
            go.Histogram(x=df['confidence'], nbinsx=20, name='Confidence'),
            row=1, col=1
        )
        
        # 2. Risk level pie chart
        risk_counts = df['risk_level'].value_counts()
        fig.add_trace(
            go.Pie(values=risk_counts.values, labels=risk_counts.index, name='Risk Levels'),
            row=1, col=2
        )
        
        # 3. PCA scatter
        colors = ['#4CAF50' if pred == 'benign' else '#F44336' for pred in df['predicted']]
        fig.add_trace(
            go.Scatter(
                x=df['PCA1'], y=df['PCA2'],
                mode='markers',
                marker=dict(color=colors, size=6),
                name='PCA Projection'
            ),
            row=1, col=3
        )
        
        # 4. Feature correlation (simplified)
        key_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
        corr_matrix = df[key_features].corr()
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values, x=key_features, y=key_features, colorscale='RdBu'),
            row=2, col=1
        )
        
        # 5. Prediction accuracy by confidence bins
        confidence_bins = pd.cut(df['confidence'], bins=5)
        accuracy_by_conf = df.groupby(confidence_bins)['is_correct'].mean()
        fig.add_trace(
            go.Bar(x=[str(x) for x in accuracy_by_conf.index], y=accuracy_by_conf.values, name='Accuracy'),
            row=2, col=2
        )
        
        # 6. t-SNE if available
        if 'TSNE1' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['TSNE1'], y=df['TSNE2'],
                    mode='markers',
                    marker=dict(color=colors, size=4),
                    name='t-SNE'
                ),
                row=2, col=3
            )
        
        # 7. Biomarker boxplots
        for i, diagnosis in enumerate(['benign', 'malignant']):
            subset = df[df['predicted'] == diagnosis]
            fig.add_trace(
                go.Box(y=subset['mean radius'], name=f'{diagnosis} radius', boxpoints='outliers'),
                row=3, col=1
            )
        
        fig.update_layout(
            height=1200,
            title_text="🏥 Comprehensive Medical Analytics Dashboard",
            showlegend=False
        )
        
        return fig
    
    def create_temporal_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create temporal analysis visualization (simulated)"""
        
        # Simulate temporal data
        dates = pd.date_range('2024-01-01', periods=len(df), freq='D')
        df_temporal = df.copy()
        df_temporal['date'] = dates
        df_temporal['cumulative_cases'] = range(1, len(df) + 1)
        df_temporal['cumulative_malignant'] = (df_temporal['predicted'] == 'malignant').cumsum()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Case Analysis', 'Cumulative Malignant Detection Rate'),
            shared_xaxes=True
        )
        
        # Daily cases by diagnosis
        for diagnosis in ['benign', 'malignant']:
            daily_counts = df_temporal[df_temporal['predicted'] == diagnosis].groupby('date').size()
            fig.add_trace(
                go.Scatter(
                    x=daily_counts.index,
                    y=daily_counts.values,
                    name=f'Daily {diagnosis.title()}',
                    mode='lines+markers'
                ),
                row=1, col=1
            )
        
        # Cumulative detection rate
        detection_rate = (df_temporal['cumulative_malignant'] / df_temporal['cumulative_cases']) * 100
        fig.add_trace(
            go.Scatter(
                x=df_temporal['date'],
                y=detection_rate,
                name='Malignancy Detection Rate (%)',
                mode='lines',
                line=dict(color='red', width=3)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            title='📈 Temporal Diagnostic Analysis',
            xaxis_title='Date'
        )
        
        return fig

class MedicalImageSimulator:
    """Simulate medical imaging visualization"""
    
    def create_simulated_mammogram(self, features: Dict[str, float]) -> go.Figure:
        """Create simulated mammogram visualization based on features"""
        
        # Create simulated tissue background
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        X, Y = np.meshgrid(x, y)
        
        # Base tissue pattern
        tissue = np.sin(X/10) * np.cos(Y/10) + np.random.normal(0, 0.1, (100, 100))
        
        # Add mass based on features
        center_x, center_y = 50, 50
        radius = features.get('mean radius', 10)
        area = features.get('mean area', 500)
        
        # Create mass
        mass_intensity = 1.5 if features.get('worst concavity', 0) > 0.1 else 1.0
        for i in range(100):
            for j in range(100):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if dist <= radius:
                    tissue[i, j] += mass_intensity * np.exp(-dist/radius)
        
        fig = go.Figure(data=go.Heatmap(
            z=tissue,
            colorscale='Greys',
            showscale=False,
            hovertemplate='Tissue Density: %{z:.2f}<extra></extra>'
        ))
        
        # Add annotations
        annotations = []
        if features.get('worst concavity', 0) > 0.15:
            annotations.append(dict(
                x=center_x, y=center_y,
                text="⚠️ Irregular\nContours",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                font=dict(color="red", size=12)
            ))
        
        fig.update_layout(
            title='📷 Simulated Mammographic View',
            annotations=annotations,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=500,
            width=500
        )
        
        return fig

def create_medical_summary_card(patient_data: Dict) -> str:
    """Create HTML summary card for patient"""
    
    card_html = f"""
    <div style="border: 2px solid #ddd; border-radius: 10px; padding: 20px; margin: 10px 0; 
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);">
        <h3 style="color: #2c3e50; margin-top: 0;">🏥 Patient Summary</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div>
                <p><strong>📊 Diagnosis:</strong> {patient_data.get('predicted', 'N/A').upper()}</p>
                <p><strong>🎯 Confidence:</strong> {patient_data.get('confidence', 0):.1%}</p>
                <p><strong>⚠️ Risk Level:</strong> {patient_data.get('risk_level', 'Unknown')}</p>
            </div>
            <div>
                <p><strong>📏 Mean Radius:</strong> {patient_data.get('mean radius', 0):.2f}</p>
                <p><strong>📐 Mean Area:</strong> {patient_data.get('mean area', 0):.0f}</p>
                <p><strong>🔍 Texture:</strong> {patient_data.get('mean texture', 0):.2f}</p>
            </div>
        </div>
    </div>
    """
    
    return card_html
