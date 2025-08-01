#!/bin/bash

# MedAI Diagnostic Suite - Quick Start Script
# ==========================================

echo "🏥 Setting up MedAI Advanced Diagnostic Suite..."
echo "================================================="

# Check Python version
python_version=$(python --version 2>&1)
echo "📋 Python Version: $python_version"

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Verify model file exists
if [ -f "breast_cancer_rf_model.joblib" ]; then
    echo "✅ Model file found: breast_cancer_rf_model.joblib"
else
    echo "⚠️  Model file not found. Running model training..."
    python -c "
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from joblib import dump

print('🔄 Training diagnostic model...')
data = load_breast_cancer()
X, y = data.data, data.target
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
dump(clf, 'breast_cancer_rf_model.joblib')
print('✅ Model trained and saved!')
"
fi

# Create startup scripts
cat > start_basic_dashboard.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Basic Diagnostic Dashboard..."
streamlit run diagnostic_dashboard.py --server.port 8501
EOF

cat > start_enhanced_dashboard.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Enhanced MedAI Diagnostic Suite..."
streamlit run enhanced_dashboard.py --server.port 8502
EOF

cat > start_basic_dashboard.bat << 'EOF'
@echo off
echo 🚀 Starting Basic Diagnostic Dashboard...
streamlit run diagnostic_dashboard.py --server.port 8501
EOF

cat > start_enhanced_dashboard.bat << 'EOF'
@echo off
echo 🚀 Starting Enhanced MedAI Diagnostic Suite...
streamlit run enhanced_dashboard.py --server.port 8502
EOF

chmod +x start_basic_dashboard.sh
chmod +x start_enhanced_dashboard.sh

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "🚀 Quick Start Options:"
echo "1. Basic Dashboard:    streamlit run diagnostic_dashboard.py"
echo "2. Enhanced Suite:     streamlit run enhanced_dashboard.py" 
echo "3. Use startup scripts for convenience"
echo ""
echo "🌐 Default URLs:"
echo "- Basic:     http://localhost:8501"
echo "- Enhanced:  http://localhost:8502"
echo ""
echo "📚 Features Available:"
echo "- 🏥 Patient Analysis with 3D Visualizations"
echo "- 🧬 Molecular Pattern Analysis" 
echo "- ⚠️  Risk Assessment Center"
echo "- 📊 Advanced Analytics Hub"
echo "- 🎯 AI Prediction Laboratory"
echo "- 🔴 Live Monitoring (Simulation)"
echo "- 📋 Clinical Report Generation"
echo ""
echo "💡 For help: streamlit run enhanced_dashboard.py --help"
echo "📖 Documentation: See README.md for detailed usage"
