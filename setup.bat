@echo off
REM MedAI Diagnostic Suite - Windows Setup Script
REM ===============================================

echo 🏥 Setting up MedAI Advanced Diagnostic Suite...
echo =================================================

REM Check Python version
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Install requirements
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Check for model file
if exist "breast_cancer_rf_model.joblib" (
    echo ✅ Model file found: breast_cancer_rf_model.joblib
) else (
    echo ⚠️  Model file not found. Training model...
    python -c "from sklearn.ensemble import RandomForestClassifier; from sklearn.datasets import load_breast_cancer; from joblib import dump; print('🔄 Training diagnostic model...'); data = load_breast_cancer(); X, y = data.data, data.target; clf = RandomForestClassifier(n_estimators=100, random_state=42); clf.fit(X, y); dump(clf, 'breast_cancer_rf_model.joblib'); print('✅ Model trained and saved!')"
)

echo.
echo 🎉 Setup Complete!
echo ==================
echo.
echo 🚀 Quick Start Commands:
echo 1. Basic Dashboard:    streamlit run diagnostic_dashboard.py
echo 2. Enhanced Suite:     streamlit run enhanced_dashboard.py
echo.
echo 🌐 Default URLs:
echo - Basic:     http://localhost:8501
echo - Enhanced:  http://localhost:8502
echo.
echo 📚 Available Features:
echo - 🏥 Patient Analysis with 3D Visualizations
echo - 🧬 Molecular Pattern Analysis
echo - ⚠️  Risk Assessment Center
echo - 📊 Advanced Analytics Hub
echo - 🎯 AI Prediction Laboratory
echo - 🔴 Live Monitoring Simulation
echo - 📋 Clinical Report Generation
echo.
echo 💡 Press any key to start Enhanced Dashboard...
pause
streamlit run enhanced_dashboard.py
