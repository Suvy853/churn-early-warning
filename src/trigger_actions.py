# src/trigger_actions.py
"""
Trigger Action Layer

This script is called by the scheduler after predictions are made.
It sends alerts for high-risk customers and logs interventions.

Usage: python src/trigger_actions.py
"""

from actions import trigger_all_actions
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    try:
        logger.info("Starting Action Layer trigger...")
        trigger_all_actions()
        logger.info("✓ Action Layer completed successfully")
    except Exception as e:
        logger.error(f"✗ Error in Action Layer: {e}")
        raise