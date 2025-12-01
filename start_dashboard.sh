#!/bin/bash
# Simple script to start the Streamlit dashboard

echo "🚀 Starting Aviation Analytics Dashboard..."
echo ""
echo "The dashboard will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""
cd "$(dirname "$0")"
streamlit run dashboard.py

