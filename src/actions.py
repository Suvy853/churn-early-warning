# src/actions.py
"""
Action Layer - Automated Interventions and Alerts

Generates alerts for high-risk customers and logs interventions.
Sends simulated emails (no actual email server needed for demo).
"""

import sqlite3
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_connection(db_path='data/churn_system.db'):
    """Get database connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def create_interventions_table(db_path='data/churn_system.db'):
    """Create interventions tracking table if it doesn't exist."""
    
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS interventions (
            intervention_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'sent',
            recipient_email TEXT,
            message_body TEXT,
            notes TEXT
        )
        ''')
        
        conn.commit()
        logger.info("✓ Interventions table created/verified")
    
    except Exception as e:
        logger.error(f"Error creating interventions table: {e}")
    
    finally:
        conn.close()


def get_high_risk_customers(db_path='data/churn_system.db'):
    """Get all high-risk customers that haven't been alerted yet."""
    
    conn = get_db_connection(db_path)
    
    query = '''
    SELECT DISTINCT p.customer_id, p.churn_probability, p.risk_tier, 
                    p.monthly_revenue, p.annual_revenue_at_risk, 
                    p.health_score, p.recommendation
    FROM predictions p
    WHERE p.prediction_date = (SELECT MAX(prediction_date) FROM predictions)
    AND p.risk_tier = 'High Risk'
    AND p.customer_id NOT IN (
        SELECT DISTINCT customer_id FROM interventions 
        WHERE DATE(timestamp) = DATE('now')
    )
    ORDER BY p.annual_revenue_at_risk DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


def generate_email_alert(customer_id, churn_prob, recommendation, revenue_at_risk):
    """Generate alert email for high-risk customer."""
    
    email_subject = f"URGENT: Customer {customer_id} At Risk of Churn"
    
    email_body = f"""
    CUSTOMER CHURN ALERT
    {'='*60}
    
    Customer ID: {customer_id}
    Churn Probability: {churn_prob*100:.1f}%
    Risk Level: HIGH RISK
    Annual Revenue at Risk: ${revenue_at_risk:,.2f}
    
    RECOMMENDED ACTION:
    {recommendation}
    
    ACTION REQUIRED:
    1. Review customer account immediately
    2. Contact customer within 24 hours
    3. Implement recommended retention strategy
    4. Log outcome in system
    
    Timeline: This alert was triggered by our predictive model.
    Immediate action recommended.
    
    {'='*60}
    Sent: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    return email_subject, email_body


def send_alert_email(customer_id, email_subject, email_body, db_path='data/churn_system.db'):
    """Simulate sending alert email and log intervention."""
    
    # In production, this would use SMTP to send real emails
    # For demo, we simulate sending and log to database
    
    recipient_email = f"sales-team+{customer_id}@company.com"
    
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO interventions 
        (customer_id, action_type, status, recipient_email, message_body, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            customer_id,
            'email_alert',
            'sent',
            recipient_email,
            email_body,
            f"Subject: {email_subject}"
        ))
        
        conn.commit()
        intervention_id = cursor.lastrowid
        conn.close()
        
        logger.info(f"✓ Alert sent for {customer_id} (ID: {intervention_id})")
        
        return True, intervention_id
    
    except Exception as e:
        logger.error(f"Error sending alert for {customer_id}: {e}")
        return False, None


def create_retention_flag(customer_id, db_path='data/churn_system.db'):
    """Create retention flag for high-risk customer."""
    
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO interventions 
        (customer_id, action_type, status, notes)
        VALUES (?, ?, ?, ?)
        ''', (
            customer_id,
            'retention_flag_created',
            'active',
            'Flagged for retention team review'
        ))
        
        conn.commit()
        intervention_id = cursor.lastrowid
        conn.close()
        
        logger.info(f"✓ Retention flag created for {customer_id}")
        
        return True, intervention_id
    
    except Exception as e:
        logger.error(f"Error creating retention flag for {customer_id}: {e}")
        return False, None


def get_intervention_stats(db_path='data/churn_system.db'):
    """Get intervention statistics."""
    
    conn = get_db_connection(db_path)
    
    query = '''
    SELECT 
        COUNT(*) as total_interventions,
        COUNT(DISTINCT customer_id) as unique_customers,
        SUM(CASE WHEN action_type = 'email_alert' THEN 1 ELSE 0 END) as emails_sent,
        SUM(CASE WHEN action_type = 'retention_flag_created' THEN 1 ELSE 0 END) as flags_created,
        SUM(CASE WHEN DATE(timestamp) = DATE('now') THEN 1 ELSE 0 END) as today_interventions
    FROM interventions
    '''
    
    result = pd.read_sql_query(query, conn)
    conn.close()
    
    return result.iloc[0].to_dict() if not result.empty else {}


def get_intervention_history(customer_id, db_path='data/churn_system.db'):
    """Get intervention history for a specific customer."""
    
    conn = get_db_connection(db_path)
    
    query = '''
    SELECT intervention_id, customer_id, action_type, timestamp, status, notes
    FROM interventions
    WHERE customer_id = ?
    ORDER BY timestamp DESC
    '''
    
    df = pd.read_sql_query(query, conn, params=(customer_id,))
    conn.close()
    
    return df


def trigger_all_actions(db_path='data/churn_system.db'):
    """
    Main function: Trigger all interventions for high-risk customers.
    Called daily by scheduler.
    """
    
    print("\n" + "=" * 70)
    print("ACTION LAYER - TRIGGERING INTERVENTIONS")
    print("=" * 70)
    
    # Ensure interventions table exists
    create_interventions_table(db_path)
    
    # Get high-risk customers
    high_risk = get_high_risk_customers(db_path)
    
    if high_risk.empty:
        print("\nNo new high-risk customers to alert.")
        return
    
    print(f"\nFound {len(high_risk)} high-risk customers needing intervention.")
    
    alerts_sent = 0
    flags_created = 0
    
    # Process each high-risk customer
    for _, customer in high_risk.iterrows():
        customer_id = customer['customer_id']
        churn_prob = customer['churn_probability']
        recommendation = customer['recommendation']
        revenue_at_risk = customer['annual_revenue_at_risk']
        
        print(f"\n→ Processing {customer_id} (Churn: {churn_prob*100:.1f}%)")
        
        # Step 1: Generate and send email alert
        subject, body = generate_email_alert(
            customer_id, churn_prob, recommendation, revenue_at_risk
        )
        
        success, intervention_id = send_alert_email(
            customer_id, subject, body, db_path
        )
        
        if success:
            alerts_sent += 1
        
        # Step 2: Create retention flag
        success, flag_id = create_retention_flag(customer_id, db_path)
        
        if success:
            flags_created += 1
    
    # Print summary
    print("\n" + "=" * 70)
    print("ACTION LAYER SUMMARY")
    print("=" * 70)
    print(f"Total customers processed: {len(high_risk)}")
    print(f"Alerts sent: {alerts_sent}")
    print(f"Retention flags created: {flags_created}")
    
    # Get overall stats
    stats = get_intervention_stats(db_path)
    print(f"\nOverall Statistics:")
    print(f"  Total interventions: {stats.get('total_interventions', 0)}")
    print(f"  Unique customers: {stats.get('unique_customers', 0)}")
    print(f"  Total emails sent: {stats.get('emails_sent', 0)}")
    print(f"  Total flags created: {stats.get('flags_created', 0)}")
    print(f"  Today's interventions: {stats.get('today_interventions', 0)}")
    print("\n✓ Action Layer complete!")


if __name__ == "__main__":
    trigger_all_actions()