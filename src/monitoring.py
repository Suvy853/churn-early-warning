# src/monitoring.py
"""
Model Monitoring System

Tracks model health and detects prediction drift.
Monitors changes in predictions, features, and model performance over time.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_connection(db_path='data/churn_system.db'):
    """Get database connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def create_monitoring_table(db_path='data/churn_system.db'):
    """Create model monitoring metrics table."""
    
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_date DATE DEFAULT CURRENT_DATE,
            metric_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            metric_type TEXT,
            metric_name TEXT,
            metric_value REAL,
            status TEXT DEFAULT 'normal',
            alert_flag BOOLEAN DEFAULT 0,
            notes TEXT
        )
        ''')
        
        conn.commit()
        logger.info("✓ Monitoring metrics table created/verified")
    
    except Exception as e:
        logger.error(f"Error creating monitoring table: {e}")
    
    finally:
        conn.close()


def get_prediction_statistics(db_path='data/churn_system.db'):
    """Get prediction statistics for latest prediction run."""
    
    conn = get_db_connection(db_path)
    
    query = '''
    SELECT 
        prediction_date,
        COUNT(*) as total_predictions,
        AVG(churn_probability) as avg_churn,
        MIN(churn_probability) as min_churn,
        MAX(churn_probability) as max_churn,
        SUM(CASE WHEN risk_tier = 'High Risk' THEN 1 ELSE 0 END) as high_risk_count,
        SUM(CASE WHEN risk_tier = 'Medium Risk' THEN 1 ELSE 0 END) as medium_risk_count,
        SUM(CASE WHEN risk_tier = 'Low Risk' THEN 1 ELSE 0 END) as low_risk_count,
        SUM(annual_revenue_at_risk) as total_revenue_at_risk
    FROM predictions
    GROUP BY prediction_date
    ORDER BY prediction_date DESC
    LIMIT 1
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


def detect_prediction_drift(db_path='data/churn_system.db', threshold=0.15):
    """
    Detect if prediction distribution has changed significantly.
    
    Compares latest prediction run with previous one.
    If change > threshold, flag as drift.
    """
    
    conn = get_db_connection(db_path)
    
    # Get latest two prediction dates
    dates_query = '''
    SELECT DISTINCT prediction_date FROM predictions
    ORDER BY prediction_date DESC
    LIMIT 2
    '''
    dates = pd.read_sql_query(dates_query, conn)
    
    if len(dates) < 2:
        conn.close()
        return False, 0, "Insufficient historical data (need 2+ runs)"
    
    latest_date = dates.iloc[0]['prediction_date']
    previous_date = dates.iloc[1]['prediction_date']
    
    # Get average churn for latest run
    latest_query = '''
    SELECT AVG(churn_probability) as avg_churn
    FROM predictions
    WHERE prediction_date = ?
    '''
    latest_avg = pd.read_sql_query(latest_query, conn, params=(latest_date,)).iloc[0]['avg_churn']
    
    # Get average churn for previous run
    previous_query = '''
    SELECT AVG(churn_probability) as avg_churn
    FROM predictions
    WHERE prediction_date = ?
    '''
    previous_avg = pd.read_sql_query(previous_query, conn, params=(previous_date,)).iloc[0]['avg_churn']
    
    conn.close()
    
    if pd.isna(latest_avg) or pd.isna(previous_avg) or previous_avg == 0:
        return False, 0, "Insufficient data for comparison"
    
    # Calculate percentage change
    pct_change = abs((latest_avg - previous_avg) / previous_avg)
    
    is_drift = pct_change > threshold
    
    return is_drift, pct_change, f"Change: {pct_change*100:.2f}%"


def detect_risk_distribution_anomaly(db_path='data/churn_system.db'):
    """
    Detect if risk distribution is abnormal.
    
    Normal: Low ~93%, Medium ~6%, High ~1%
    Alert if High Risk > 5% or Medium Risk > 15%
    """
    
    conn = get_db_connection(db_path)
    
    query = '''
    SELECT 
        SUM(CASE WHEN risk_tier = 'High Risk' THEN 1 ELSE 0 END) as high_risk,
        SUM(CASE WHEN risk_tier = 'Medium Risk' THEN 1 ELSE 0 END) as medium_risk,
        SUM(CASE WHEN risk_tier = 'Low Risk' THEN 1 ELSE 0 END) as low_risk,
        COUNT(*) as total
    FROM predictions
    WHERE prediction_date = (SELECT MAX(prediction_date) FROM predictions)
    '''
    
    result = pd.read_sql_query(query, conn).iloc[0]
    conn.close()
    
    total = result['total']
    high_risk = result['high_risk'] if result['high_risk'] else 0
    medium_risk = result['medium_risk'] if result['medium_risk'] else 0
    
    high_pct = high_risk / total if total > 0 else 0
    medium_pct = medium_risk / total if total > 0 else 0
    
    anomalies = []
    
    if high_pct > 0.05:
        anomalies.append(f"High Risk > 5% (actual: {high_pct*100:.1f}%)")
    
    if medium_pct > 0.15:
        anomalies.append(f"Medium Risk > 15% (actual: {medium_pct*100:.1f}%)")
    
    is_anomaly = len(anomalies) > 0
    message = "; ".join(anomalies) if anomalies else "Normal distribution"
    
    return is_anomaly, high_pct, medium_pct, message


def log_metric(metric_type, metric_name, metric_value, status='normal', 
               alert_flag=False, notes='', db_path='data/churn_system.db'):
    """Log a monitoring metric to the database."""
    
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO model_metrics 
        (metric_type, metric_name, metric_value, status, alert_flag, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (metric_type, metric_name, metric_value, status, alert_flag, notes))
        
        conn.commit()
        logger.info(f"✓ Logged metric: {metric_name} = {metric_value:.4f}")
    
    except Exception as e:
        logger.error(f"Error logging metric: {e}")
    
    finally:
        conn.close()


def run_monitoring_checks(db_path='data/churn_system.db'):
    """
    Run all monitoring checks and log results.
    Called daily by scheduler.
    """
    
    print("\n" + "=" * 70)
    print("MODEL MONITORING - RUNNING HEALTH CHECKS")
    print("=" * 70)
    
    # Ensure monitoring table exists
    create_monitoring_table(db_path)
    
    alerts_triggered = 0
    
    # Check 1: Prediction Drift Detection
    print("\n→ Checking for prediction drift...")
    is_drift, pct_change, message = detect_prediction_drift(db_path, threshold=0.15)
    
    status = 'alert' if is_drift else 'normal'
    log_metric(
        'prediction_drift',
        'daily_drift_detection',
        pct_change,
        status=status,
        alert_flag=is_drift,
        notes=message,
        db_path=db_path
    )
    
    if is_drift:
        print(f"  ⚠ DRIFT ALERT: {message}")
        alerts_triggered += 1
    else:
        print(f"  ✓ Normal: {message}")
    
    # Check 2: Risk Distribution Anomaly
    print("\n→ Checking risk distribution...")
    is_anomaly, high_pct, medium_pct, message = detect_risk_distribution_anomaly(db_path)
    
    status = 'alert' if is_anomaly else 'normal'
    log_metric(
        'risk_distribution',
        'high_risk_percentage',
        high_pct,
        status=status,
        alert_flag=is_anomaly,
        notes=f"High: {high_pct*100:.1f}%, Medium: {medium_pct*100:.1f}%",
        db_path=db_path
    )
    
    if is_anomaly:
        print(f"  ⚠ ANOMALY ALERT: {message}")
        alerts_triggered += 1
    else:
        print(f"  ✓ Normal distribution: {message}")
    
    # Check 3: Prediction Statistics
    print("\n→ Recording prediction statistics...")
    stats = get_prediction_statistics(db_path)
    
    if not stats.empty:
        today_stats = stats.iloc[0]
        
        log_metric(
            'prediction_stats',
            'average_churn_probability',
            today_stats['avg_churn'],
            notes=f"Min: {today_stats['min_churn']:.4f}, Max: {today_stats['max_churn']:.4f}",
            db_path=db_path
        )
        
        log_metric(
            'prediction_stats',
            'total_revenue_at_risk',
            today_stats['total_revenue_at_risk'],
            notes=f"High Risk: {int(today_stats['high_risk_count'])}, Medium: {int(today_stats['medium_risk_count'])}",
            db_path=db_path
        )
        
        print(f"  ✓ Avg churn: {today_stats['avg_churn']:.4f}")
        print(f"  ✓ High risk customers: {int(today_stats['high_risk_count'])}")
        print(f"  ✓ Revenue at risk: ${today_stats['total_revenue_at_risk']:,.2f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("MONITORING SUMMARY")
    print("=" * 70)
    print(f"Total alerts triggered: {alerts_triggered}")
    print(f"Model status: {'⚠ ALERT' if alerts_triggered > 0 else '✓ HEALTHY'}")
    print("✓ Monitoring complete!")
    
    return alerts_triggered


if __name__ == "__main__":
    run_monitoring_checks()