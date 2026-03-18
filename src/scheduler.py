# src/scheduler.py
"""
Scheduled Pipeline Runner

Automates data ingestion, feature engineering, and scoring.
Runs daily (default: 2 AM) to update predictions.
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import score_customers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline():
    """
    Execute full pipeline:
    1. Score customers
    2. Store results
    """
    
    print("\n" + "=" * 70)
    print(f"PIPELINE RUN: {datetime.now()}")
    print("=" * 70)
    
    try:
        # Score customers
        predictions = score_customers()
        
        logger.info(f"Pipeline completed successfully. Scored {len(predictions):,} customers.")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return False


def start_scheduler(run_time='02:00'):
    """
    Start background scheduler.
    
    Args:
        run_time: Time to run daily (HH:MM format, 24-hour)
    """
    
    print("\n" + "=" * 70)
    print("SCHEDULER STARTING")
    print("=" * 70)
    
    scheduler = BackgroundScheduler()
    
    # Parse time
    hour, minute = map(int, run_time.split(':'))
    
    # Schedule job
    scheduler.add_job(
        run_pipeline,
        trigger=CronTrigger(hour=hour, minute=minute),
        id='churn_pipeline',
        name='Daily churn prediction pipeline',
        replace_existing=True
    )
    
    scheduler.start()
    
    print(f"\n✓ Scheduler started")
    print(f"  Job: Churn prediction pipeline")
    print(f"  Schedule: Daily at {run_time}")
    print(f"  Status: Running in background")
    print(f"\nPress Ctrl+C to stop scheduler")
    
    try:
        # Keep scheduler running
        while True:
            pass
    except KeyboardInterrupt:
        print("\n✓ Scheduler stopped")
        scheduler.shutdown()


def run_once():
    """Run pipeline once (useful for testing)."""
    return run_pipeline()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'once':
        # Run once for testing
        print("Running pipeline once...")
        run_once()
    else:
        # Start scheduler
        start_scheduler(run_time='02:00')