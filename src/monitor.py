# src/monitor.py
"""
Pipeline Monitoring

Check pipeline status and predictions.
"""

import sqlite3
from datetime import datetime

def get_latest_predictions(db_path='data/churn_system.db'):
    """Get latest predictions from database."""
    
    conn = sqlite3.connect(db_path)
    
    query = '''
    SELECT 
        prediction_date,
        COUNT(*) as customer_count,
        SUM(CASE WHEN risk_tier = 'High Risk' THEN 1 ELSE 0 END) as high_risk_count,
        ROUND(AVG(churn_probability), 4) as avg_churn_prob,
        ROUND(SUM(revenue_at_risk), 0) as total_revenue_at_risk
    FROM predictions
    GROUP BY prediction_date
    ORDER BY prediction_date DESC
    LIMIT 5
    '''
    
    results = conn.execute(query).fetchall()
    conn.close()
    
    return results


def print_pipeline_status():
    """Print latest pipeline status."""
    
    print("\n" + "=" * 70)
    print("PIPELINE STATUS")
    print("=" * 70)
    
    results = get_latest_predictions()
    
    if not results:
        print("\nNo predictions found in database.")
        return
    
    print("\nLatest Prediction Runs:")
    print(f"{'Date':<15} {'Customers':<12} {'High Risk':<12} {'Avg Churn %':<12} {'Revenue@Risk':<15}")
    print("-" * 70)
    
    for row in results:
        date, count, high_risk, avg_prob, revenue = row
        print(f"{date:<15} {count:<12} {high_risk:<12} {avg_prob*100:<11.1f}% ${revenue:<14,.0f}")
    
    # Latest details
    latest = results[0]
    print(f"\n✓ Latest run: {latest[0]}")
    print(f"  Customers: {latest[1]:,}")
    print(f"  High Risk: {latest[2]:,} ({latest[2]/latest[1]*100:.1f}%)")
    print(f"  Revenue at Risk: ${latest[4]:,.0f}")


if __name__ == "__main__":
    print_pipeline_status()