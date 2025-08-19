#!/bin/bash

echo "🚀 Starting Olist E-Commerce Analytics Platform..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Run Streamlit app
echo "🌐 Launching Streamlit app..."
echo "📊 Dashboard will open at: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

streamlit run app.py --server.port 8501 --server.address localhost