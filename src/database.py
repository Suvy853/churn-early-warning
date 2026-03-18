# src/database.py
"""
Database initialization and management.

Creates SQLite database with schema for storing:
- Raw engagement data
- Engineered features
- Model predictions
- Prediction history
"""

import sqlite3
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

def init_database(db_path='data/churn_system.db'):
    """
    Initialize SQLite database with required tables.
    """
    
    print("\n" + "=" * 70)
    print("DATABASE INITIALIZATION")
    print("=" * 70)
    
    # Create database if doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"\nInitializing database: {db_path}")
    
    # Table 1: Customer metadata
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY,
        company_name TEXT,
        company_size TEXT,
        subscription_tier TEXT,
        industry TEXT,
        monthly_revenue REAL,
        onboarding_date TEXT,
        contract_length_months INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    print("✓ Created customers table")
    
    # Table 2: Raw engagement data
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS engagement_raw (
        engagement_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER,
        year_month TEXT,
        active_users INTEGER,
        api_calls INTEGER,
        logins_per_day REAL,
        days_since_last_login INTEGER,
        features_used_count INTEGER,
        support_tickets INTEGER,
        engagement_score REAL,
        churned INTEGER,
        ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
    ''')
    print("✓ Created engagement_raw table")
    
    # Table 3: Model predictions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER,
        prediction_date TEXT,
        churn_probability REAL,
        predicted_churn INTEGER,
        risk_tier TEXT,
        health_score REAL,
        monthly_revenue REAL,
        revenue_at_risk REAL,
        annual_revenue_at_risk REAL,
        composite_risk_score REAL,
        recommendation TEXT,
        risk_rank INTEGER,
        predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
    ''')
    print("✓ Created predictions table")
    
    # Create indices for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_customer_id ON customers(customer_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_engagement_customer ON engagement_raw(customer_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_customer ON predictions(customer_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date)')
    print("✓ Created indices")
    
    conn.commit()
    conn.close()
    
    print(f"\n✓ Database initialized successfully!")
    return db_path


def get_connection(db_path='data/churn_system.db'):
    """Get database connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


if __name__ == "__main__":
    init_database()