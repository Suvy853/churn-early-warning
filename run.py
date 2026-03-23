#!/usr/bin/env python
"""
Startup script for Churn Early Warning System

Runs Dash dashboard on port 8050
FastAPI is available for local development via: python src/api.py
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def start_dashboard():
    """Start Dash dashboard."""
    print("\n" + "=" * 70)
    print("CHURN EARLY WARNING SYSTEM - DASHBOARD")
    print("=" * 70)
    print("\nStarting Dash Dashboard on port 8050...")
    print("Access at: http://127.0.0.1:8050\n")
    
    from src.app import app
    app.run(debug=False, host='0.0.0.0', port=8050)


if __name__ == "__main__":
    start_dashboard()