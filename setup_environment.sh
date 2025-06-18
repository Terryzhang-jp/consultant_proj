#!/bin/bash

# POJ Hypothesis Testing Environment Setup Script
# This script creates a virtual environment and installs all required dependencies

echo "🚀 Setting up POJ Hypothesis Testing Environment..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing required packages..."
pip install -r requirements.txt

# Verify installation
echo "✅ Verifying installation..."
python -c "
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
print('✅ All core packages installed successfully!')
print(f'   - pandas: {pd.__version__}')
print(f'   - numpy: {np.__version__}')
print(f'   - scikit-learn: {sklearn.__version__}')
print(f'   - xgboost: {xgb.__version__}')
print(f'   - shap: {shap.__version__}')
"

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "   source venv/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "   deactivate"
echo ""
echo "To run hypothesis tests:"
echo "   cd hypothesis_testing/hypothesis_X"
echo "   python run.py"
echo ""
