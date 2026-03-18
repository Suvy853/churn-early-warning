# src/api_actions.py
"""
FastAPI endpoints for Action Layer

Exposes intervention data through REST API.
Add these endpoints to your existing api.py
"""

from fastapi import HTTPException
from pydantic import BaseModel
import sqlite3
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('data/churn_system.db')
    conn.row_factory = sqlite3.Row
    return conn


# Response Models
class Intervention(BaseModel):
    """Single intervention record."""
    intervention_id: int
    customer_id: str
    action_type: str
    timestamp: str
    status: str
    recipient_email: str = None
    notes: str = None
    
    class Config:
        example = {
            "intervention_id": 1,
            "customer_id": "CUST_0001",
            "action_type": "email_alert",
            "timestamp": "2026-03-18 14:30:00",
            "status": "sent",
            "recipient_email": "sales-team+CUST_0001@company.com",
            "notes": "Subject: URGENT: Customer CUST_0001 At Risk of Churn"
        }


class InterventionStats(BaseModel):
    """Intervention statistics."""
    total_interventions: int
    unique_customers: int
    emails_sent: int
    flags_created: int
    today_interventions: int
    
    class Config:
        example = {
            "total_interventions": 127,
            "unique_customers": 45,
            "emails_sent": 89,
            "flags_created": 38,
            "today_interventions": 12
        }


# Endpoints for Action Layer

def get_interventions(limit: int = 100):
    """
    Get recent interventions.
    
    Returns: List of recent alerts and actions sent.
    """
    
    try:
        conn = get_db_connection()
        
        query = '''
        SELECT intervention_id, customer_id, action_type, timestamp, 
               status, recipient_email, notes
        FROM interventions
        ORDER BY timestamp DESC
        LIMIT ?
        '''
        
        results = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        if results.empty:
            return {"total": 0, "interventions": []}
        
        interventions = []
        for _, row in results.iterrows():
            interventions.append({
                "intervention_id": int(row['intervention_id']),
                "customer_id": row['customer_id'],
                "action_type": row['action_type'],
                "timestamp": row['timestamp'],
                "status": row['status'],
                "recipient_email": row['recipient_email'],
                "notes": row['notes']
            })
        
        return {
            "total": len(interventions),
            "interventions": interventions
        }
    
    except Exception as e:
        logger.error(f"Error getting interventions: {e}")
        raise HTTPException(status_code=500, detail="Error getting interventions")


def get_customer_interventions(customer_id: str):
    """
    Get all interventions for a specific customer.
    
    Returns: Complete intervention history for customer.
    """
    
    try:
        conn = get_db_connection()
        
        query = '''
        SELECT intervention_id, customer_id, action_type, timestamp, 
               status, recipient_email, notes
        FROM interventions
        WHERE customer_id = ?
        ORDER BY timestamp DESC
        '''
        
        results = pd.read_sql_query(query, conn, params=(customer_id,))
        conn.close()
        
        if results.empty:
            return {
                "customer_id": customer_id,
                "total_interventions": 0,
                "history": []
            }
        
        history = []
        for _, row in results.iterrows():
            history.append({
                "intervention_id": int(row['intervention_id']),
                "action_type": row['action_type'],
                "timestamp": row['timestamp'],
                "status": row['status'],
                "notes": row['notes']
            })
        
        return {
            "customer_id": customer_id,
            "total_interventions": len(history),
            "history": history
        }
    
    except Exception as e:
        logger.error(f"Error getting customer interventions: {e}")
        raise HTTPException(status_code=500, detail="Error getting customer interventions")


def get_intervention_statistics():
    """
    Get intervention statistics.
    
    Returns: Total interventions, emails sent, flags created, today's count.
    """
    
    try:
        conn = get_db_connection()
        
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
        
        if result.empty:
            return InterventionStats(
                total_interventions=0,
                unique_customers=0,
                emails_sent=0,
                flags_created=0,
                today_interventions=0
            )
        
        stats = result.iloc[0]
        
        return InterventionStats(
            total_interventions=int(stats['total_interventions'] or 0),
            unique_customers=int(stats['unique_customers'] or 0),
            emails_sent=int(stats['emails_sent'] or 0),
            flags_created=int(stats['flags_created'] or 0),
            today_interventions=int(stats['today_interventions'] or 0)
        )
    
    except Exception as e:
        logger.error(f"Error getting intervention statistics: {e}")
        raise HTTPException(status_code=500, detail="Error getting intervention statistics")


def get_today_interventions():
    """
    Get interventions from today.
    
    Returns: All interventions triggered today.
    """
    
    try:
        conn = get_db_connection()
        
        query = '''
        SELECT intervention_id, customer_id, action_type, timestamp, 
               status, recipient_email, notes
        FROM interventions
        WHERE DATE(timestamp) = DATE('now')
        ORDER BY timestamp DESC
        '''
        
        results = pd.read_sql_query(query, conn)
        conn.close()
        
        if results.empty:
            return {"date": str(pd.Timestamp.now().date()), "total": 0, "interventions": []}
        
        interventions = []
        for _, row in results.iterrows():
            interventions.append({
                "intervention_id": int(row['intervention_id']),
                "customer_id": row['customer_id'],
                "action_type": row['action_type'],
                "timestamp": row['timestamp'],
                "status": row['status'],
                "recipient_email": row['recipient_email'],
                "notes": row['notes']
            })
        
        return {
            "date": str(pd.Timestamp.now().date()),
            "total": len(interventions),
            "interventions": interventions
        }
    
    except Exception as e:
        logger.error(f"Error getting today interventions: {e}")
        raise HTTPException(status_code=500, detail="Error getting today interventions")


def get_alert_summary():
    """
    Get high-level summary of all alerts and interventions.
    
    Returns: Summary with key metrics and recent activity.
    """
    
    try:
        stats = get_intervention_statistics()
        today = get_today_interventions()
        
        return {
            "summary": {
                "total_interventions_all_time": stats.total_interventions,
                "unique_customers_alerted": stats.unique_customers,
                "total_emails_sent": stats.emails_sent,
                "total_flags_created": stats.flags_created
            },
            "today": {
                "date": today["date"],
                "interventions_today": today["total"],
                "recent_activity": today["interventions"][:10] if today["interventions"] else []
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting alert summary: {e}")
        raise HTTPException(status_code=500, detail="Error getting alert summary")