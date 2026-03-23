#!/usr/bin/env python
"""
Startup script for Churn Early Warning System

Runs both services concurrently:
- Dash dashboard on port 8050
- FastAPI on port 8000
"""

import threading
import time
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def start_dash():
    """Start Dash dashboard."""
    print("\n" + "="*70)
    print("STARTING DASH DASHBOARD (Port 8050)")
    print("="*70)
    
    from src.app import app
    app.run(debug=False, host='0.0.0.0', port=8050)


def start_fastapi():
    """Start FastAPI server."""
    print("\n" + "="*70)
    print("STARTING FASTAPI SERVER (Port 8000)")
    print("="*70)
    
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, log_level="info")


def main():
    """Start both services in separate threads."""
    
    print("\n" + "=" * 70)
    print("CHURN EARLY WARNING SYSTEM - STARTING ALL SERVICES")
    print("=" * 70)
    print("\nStarting services...")
    print("  • Dash Dashboard: http://127.0.0.1:8050")
    print("  • FastAPI Server: http://127.0.0.1:8000/docs")
    print("\nPress Ctrl+C to stop all services\n")
    
    # Create threads
    dash_thread = threading.Thread(target=start_dash, daemon=False)
    fastapi_thread = threading.Thread(target=start_fastapi, daemon=False)
    
    try:
        # Start both services
        fastapi_thread.start()
        time.sleep(2)  # Give FastAPI time to start
        dash_thread.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n✓ Shutting down services...")
        sys.exit(0)


if __name__ == "__main__":
    main()