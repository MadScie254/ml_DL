"""
Medical Utilities Module
========================
Advanced medical analysis utilities for breast cancer diagnosis.
Provides clinical interpretation, risk stratification, and medical insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    LOW = "Low Risk"
    MEDIUM = "Medium Risk" 
    HIGH = "High Risk"
    CRITICAL = "Critical Risk"

class DiagnosisClass(Enum):
    BENIGN = "benign"
    MALIGNANT = "malignant"

@dataclass
class PatientProfile:
    """Comprehensive patient diagnostic profile"""
    patient_id: str
    age: Optional[int] = None
    family_history: Optional[bool] = None
    biomarkers: Dict[str, float] = None
    risk_factors: List[str] = None
    previous_screenings: List[Dict] = None

@dataclass
class DiagnosticResult:
    """Complete diagnostic analysis result"""
    prediction: DiagnosisClass
    confidence: float
    risk_level: RiskLevel
    key_features: Dict[str, float]
    clinical_notes: List[str]
    recommendations: List[str]
    follow_up_timeline: str

class MedicalAnalyzer:
    """Advanced medical analysis and interpretation system"""
    
    def __init__(self):
        # Clinical thresholds based on medical literature
        self.feature_thresholds = {
            'mean radius': {'low': 10.0, 'medium': 15.0, 'high': 20.0},
            'mean texture': {'low': 10.0, 'medium': 20.0, 'high': 30.0},
            'mean perimeter': {'low': 60.0, 'medium': 90.0, 'high': 120.0},
            'mean area': {'low': 300.0, 'medium': 700.0, 'high': 1200.0},
            'worst concavity': {'low': 0.05, 'medium': 0.15, 'high': 0.25},
            'worst symmetry': {'low': 0.15, 'medium': 0.25, 'high': 0.35}
        }
        
        # Risk factor weights
        self.risk_weights = {
            'age_over_50': 1.2,
            'family_history': 1.5,
            'previous_benign_findings': 0.8,
            'hormone_therapy': 1.1,
            'smoking_history': 1.1
        }
    
    def assess_risk_level(self, confidence: float, prediction: str) -> RiskLevel:
        """Determine clinical risk level based on prediction confidence"""
        if prediction == 'malignant':
            if confidence > 0.90:
                return RiskLevel.CRITICAL
            elif confidence > 0.75:
                return RiskLevel.HIGH
            else:
                return RiskLevel.MEDIUM
        else:  # benign
            if confidence > 0.90:
                return RiskLevel.LOW
            elif confidence > 0.75:
                return RiskLevel.LOW
            else:
                return RiskLevel.MEDIUM
    
    def generate_clinical_interpretation(self, features: Dict[str, float], 
                                       prediction: str, confidence: float) -> Dict[str, any]:
        """Generate comprehensive clinical interpretation"""
        
        interpretation = {
            'biomarker_analysis': self._analyze_biomarkers(features),
            'risk_assessment': self._assess_clinical_risk(features, prediction),
            'clinical_notes': self._generate_clinical_notes(features, prediction, confidence),
            'recommendations': self._generate_recommendations(prediction, confidence),
            'follow_up': self._determine_follow_up(prediction, confidence)
        }
        
        return interpretation
    
    def _analyze_biomarkers(self, features: Dict[str, float]) -> Dict[str, str]:
        """Analyze individual biomarkers against clinical thresholds"""
        analysis = {}
        
        for feature, value in features.items():
            if feature in self.feature_thresholds:
                thresholds = self.feature_thresholds[feature]
                
                if value <= thresholds['low']:
                    level = "Within normal range"
                elif value <= thresholds['medium']:
                    level = "Slightly elevated"
                elif value <= thresholds['high']:
                    level = "Moderately elevated"
                else:
                    level = "Significantly elevated"
                
                analysis[feature] = f"{level} ({value:.2f})"
        
        return analysis
    
    def _assess_clinical_risk(self, features: Dict[str, float], prediction: str) -> Dict[str, any]:
        """Comprehensive clinical risk assessment"""
        risk_factors = []
        protective_factors = []
        
        # Analyze key morphological features
        if features.get('mean radius', 0) > 15:
            risk_factors.append("Large tumor radius")
        
        if features.get('worst concavity', 0) > 0.2:
            risk_factors.append("High concavity (irregular shape)")
        
        if features.get('worst symmetry', 0) > 0.3:
            risk_factors.append("Asymmetrical mass")
        
        if features.get('mean texture', 0) > 25:
            risk_factors.append("Heterogeneous texture")
        
        # Protective factors
        if features.get('mean radius', 0) < 12:
            protective_factors.append("Small mass size")
        
        if features.get('worst concavity', 0) < 0.1:
            protective_factors.append("Regular, smooth contours")
        
        return {
            'risk_factors': risk_factors,
            'protective_factors': protective_factors,
            'overall_risk': prediction
        }
    
    def _generate_clinical_notes(self, features: Dict[str, float], 
                               prediction: str, confidence: float) -> List[str]:
        """Generate clinical notes for medical interpretation"""
        notes = []
        
        # Confidence-based notes
        if confidence > 0.95:
            notes.append("High diagnostic confidence. Clear morphological patterns identified.")
        elif confidence > 0.80:
            notes.append("Good diagnostic confidence. Recommend correlation with clinical findings.")
        else:
            notes.append("Lower diagnostic confidence. Additional imaging or biopsy may be warranted.")
        
        # Feature-specific notes
        if features.get('mean area', 0) > 1000:
            notes.append("Large mass detected. Warrants immediate clinical attention.")
        
        if features.get('worst concavity', 0) > 0.2:
            notes.append("Irregular mass contours observed. Concerning for malignancy.")
        
        if prediction == 'malignant' and confidence > 0.85:
            notes.append("Morphological features consistent with malignant transformation.")
        
        return notes
    
    def _generate_recommendations(self, prediction: str, confidence: float) -> List[str]:
        """Generate clinical recommendations based on diagnosis"""
        recommendations = []
        
        if prediction == 'malignant':
            recommendations.extend([
                "Immediate referral to oncology",
                "Comprehensive staging workup",
                "Multidisciplinary team consultation",
                "Patient counseling and support services"
            ])
            
            if confidence < 0.80:
                recommendations.append("Consider additional confirmatory testing")
        
        else:  # benign
            if confidence > 0.90:
                recommendations.extend([
                    "Routine follow-up in 12 months",
                    "Continue regular screening schedule",
                    "Patient reassurance and education"
                ])
            else:
                recommendations.extend([
                    "Short-term follow-up in 6 months",
                    "Consider additional imaging modalities",
                    "Monitor for changes in morphology"
                ])
        
        return recommendations
    
    def _determine_follow_up(self, prediction: str, confidence: float) -> str:
        """Determine appropriate follow-up timeline"""
        if prediction == 'malignant':
            return "Immediate (within 1-2 weeks)"
        elif confidence < 0.80:
            return "Short-term (3-6 months)"
        else:
            return "Routine (12 months)"

class ClinicalReportGenerator:
    """Generate comprehensive clinical reports"""
    
    def __init__(self, analyzer: MedicalAnalyzer):
        self.analyzer = analyzer
    
    def generate_patient_report(self, patient_data: Dict, 
                              prediction_data: Dict) -> str:
        """Generate a complete patient diagnostic report"""
        
        report_template = """
BREAST CANCER DIAGNOSTIC REPORT
==============================

PATIENT INFORMATION:
Patient ID: {patient_id}
Analysis Date: {analysis_date}
Report Generated: {report_date}

DIAGNOSTIC SUMMARY:
Primary Diagnosis: {diagnosis}
Diagnostic Confidence: {confidence:.1%}
Risk Stratification: {risk_level}

MORPHOLOGICAL ANALYSIS:
{biomarker_analysis}

CLINICAL INTERPRETATION:
{clinical_notes}

RECOMMENDATIONS:
{recommendations}

FOLLOW-UP PLAN:
Next Review: {follow_up}

TECHNICAL DETAILS:
Model Used: Random Forest Classifier
Feature Vector Dimension: 30
Analysis Algorithm: Machine Learning Classification

---
This report is generated by AI-assisted diagnostic system.
All findings should be correlated with clinical assessment.
For questions, consult with radiologist or oncologist.
        """
        
        return report_template.format(**patient_data, **prediction_data)
    
    def generate_batch_summary(self, results: List[Dict]) -> str:
        """Generate summary report for batch analysis"""
        
        total_cases = len(results)
        malignant_cases = sum(1 for r in results if r['prediction'] == 'malignant')
        high_confidence = sum(1 for r in results if r['confidence'] > 0.9)
        
        summary = f"""
BATCH ANALYSIS SUMMARY REPORT
============================

Total Cases Analyzed: {total_cases}
Malignant Cases Detected: {malignant_cases} ({malignant_cases/total_cases*100:.1f}%)
Benign Cases: {total_cases - malignant_cases} ({(total_cases-malignant_cases)/total_cases*100:.1f}%)
High Confidence Predictions: {high_confidence} ({high_confidence/total_cases*100:.1f}%)

QUALITY METRICS:
- Cases requiring clinical review: {total_cases - high_confidence}
- Average confidence: {np.mean([r['confidence'] for r in results]):.1%}
- Standard deviation: {np.std([r['confidence'] for r in results]):.3f}

CLINICAL WORKFLOW RECOMMENDATIONS:
- Prioritize malignant cases for immediate review
- Schedule follow-up for low-confidence predictions
- Maintain quality assurance protocols
        """
        
        return summary

# Medical terminology and mappings
MEDICAL_TERMINOLOGY = {
    'mean radius': 'Mean Nuclear Radius',
    'mean texture': 'Mean Nuclear Texture',
    'mean perimeter': 'Mean Nuclear Perimeter', 
    'mean area': 'Mean Nuclear Area',
    'worst concavity': 'Worst Concavity Index',
    'worst symmetry': 'Worst Symmetry Score',
    'benign': 'Benign (Non-cancerous)',
    'malignant': 'Malignant (Cancerous)'
}

def get_medical_name(technical_name: str) -> str:
    """Convert technical feature names to medical terminology"""
    return MEDICAL_TERMINOLOGY.get(technical_name, technical_name.title())

def calculate_breslow_thickness_equivalent(features: Dict[str, float]) -> float:
    """
    Calculate equivalent Breslow thickness based on morphological features
    Note: This is a simplified approximation for demonstration
    """
    radius = features.get('mean radius', 0)
    area = features.get('mean area', 0)
    
    # Simplified calculation (not medically accurate - for demo only)
    thickness = (radius * 0.1) + (np.sqrt(area) * 0.05)
    return max(0.1, min(thickness, 5.0))

def generate_staging_recommendation(prediction: str, confidence: float, 
                                  features: Dict[str, float]) -> str:
    """Generate TNM staging recommendations"""
    if prediction == 'benign':
        return "N/A - Benign lesion"
    
    thickness_equiv = calculate_breslow_thickness_equivalent(features)
    
    if thickness_equiv < 1.0:
        return "Recommend T1 staging workup"
    elif thickness_equiv < 2.0:
        return "Recommend T2 staging workup"
    else:
        return "Recommend T3/T4 staging workup"
