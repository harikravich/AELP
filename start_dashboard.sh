#!/bin/bash

# GAELP Performance Dashboard Startup Script
echo "🚀 Starting GAELP Performance Dashboard"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "dashboard.py" ]; then
    echo "❌ Error: Not in GAELP project directory"
    echo "Please run this script from the AELP project root"
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

# Install requirements if needed
echo "🔧 Checking requirements..."
pip install streamlit plotly pandas matplotlib seaborn rich numpy --quiet --user

# Launch dashboard
echo "📊 Launching dashboard at http://localhost:8501"
echo "🔄 Use Ctrl+C to stop the dashboard"
echo "----------------------------------------"

python3 -m streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 --browser.gatherUsageStats false