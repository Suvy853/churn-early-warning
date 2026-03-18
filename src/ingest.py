# src/ingest.py
"""
Data Ingestion Pipeline

Loads new engagement data from source and stores in database.
"""

import pandas as pd
import sqlite3
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

def load_new_engagement_data(csv_path, db_path='data/churn_system.db'):
    """
    Load new engagement data from CSV and store in database.
    """
    
    print("\n" + "=" * 70)
    print("DATA INGESTION")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} records")
    
    # Validate
    print("\nValidating data...")
    required_cols = ['customer_id', 'year_month', 'active_users', 'api_calls', 
                    'logins_per_day', 'days_since_last_login', 'features_used_count',
                    'support_tickets', 'engagement_score', 'churned']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    print(f"✓ Data validation passed")
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=['customer_id', 'year_month'])
    after = len(df)
    print(f"✓ Removed {before - after} duplicate records")
    
    # Store in database
    conn = sqlite3.connect(db_path)
    
    print(f"\nInserting into database...")
    try:
        df.to_sql('engagement_raw', conn, if_exists='append', index=False)
        conn.commit()
        print(f"✓ Inserted {len(df):,} records into engagement_raw table")
    except Exception as e:
        conn.rollback()
        print(f"✗ Error inserting data: {e}")
        raise
    finally:
        conn.close()
    
    return len(df)


def load_customers(csv_path, db_path='data/churn_system.db'):
    """Load customer metadata."""
    
    print("\n" + "=" * 70)
    print("LOADING CUSTOMER DATA")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    print(f"Loading {len(df):,} customers...")
    
    conn = sqlite3.connect(db_path)
    
    try:
        df.to_sql('customers', conn, if_exists='replace', index=False)
        conn.commit()
        print(f"✓ Loaded {len(df):,} customers")
    except Exception as e:
        conn.rollback()
        print(f"✗ Error loading customers: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    from database import init_database
    
    print("PHASE 5: DATA INGESTION")
    print("=" * 70)
    
    # Initialize database
    init_database()
    
    # Load customer data
    load_customers('data/raw/customers.csv')
    
    # Load engagement data
    load_new_engagement_data('data/raw/engagement_monthly.csv')
    
    print("\n✓ Data ingestion complete!")