# src/predict.py
"""
Prediction and Scoring Pipeline

Scores customers with trained model and stores predictions.
"""

import pandas as pd
import numpy as np
import sqlite3
import joblib
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

def engineer_features_for_prediction(df):
    """
    Engineer features from raw engagement data.
    Same logic as Phase 2, but for real-time pipeline.
    """
    
    # Sort by customer and time
    df = df.sort_values(['customer_id', 'year_month']).reset_index(drop=True)
    
    # 3-month rolling averages
    for customer_id in df['customer_id'].unique():
        mask = df['customer_id'] == customer_id
        customer_data = df[mask].copy()
        
        df.loc[mask, 'logins_3m_avg'] = customer_data['logins_per_day'].rolling(3, min_periods=1).mean()
        df.loc[mask, 'api_calls_3m_avg'] = customer_data['api_calls'].rolling(3, min_periods=1).mean()
        df.loc[mask, 'features_3m_avg'] = customer_data['features_used_count'].rolling(3, min_periods=1).mean()
    
    # Fill NaNs
    df['logins_3m_avg'] = df['logins_3m_avg'].fillna(df['logins_per_day'])
    df['api_calls_3m_avg'] = df['api_calls_3m_avg'].fillna(df['api_calls'])
    df['features_3m_avg'] = df['features_3m_avg'].fillna(df['features_used_count'])
    
    # Trend features
    for customer_id in df['customer_id'].unique():
        mask = df['customer_id'] == customer_id
        customer_data = df[mask].copy()
        
        recent_logins = customer_data['logins_per_day'].tail(3).mean()
        previous_logins = customer_data['logins_per_day'].iloc[-6:-3].mean() if len(customer_data) >= 6 else customer_data['logins_per_day'].iloc[0]
        
        if previous_logins > 0:
            login_trend = (recent_logins - previous_logins) / previous_logins
        else:
            login_trend = 0
        
        df.loc[mask, 'logins_trend_3m'] = login_trend
    
    # Engagement decay
    df['engagement_decay'] = df.groupby('customer_id')['engagement_score'].diff()
    df['engagement_decay'] = df['engagement_decay'].fillna(0)
    
    # Recency features
    df['days_inactive'] = df['days_since_last_login']
    df['recency_score'] = 1 - (df['days_inactive'] / 60).clip(0, 1)
    
    # Adoption features
    df['feature_adoption_rate'] = df['features_used_count'] / 10.0
    df['is_high_adoption'] = (df['feature_adoption_rate'] > 0.70).astype(int)
    df['is_low_adoption'] = (df['feature_adoption_rate'] < 0.30).astype(int)
    
    # Risk flags
    df['flag_no_login_30days'] = (df['days_inactive'] >= 30).astype(int)
    df['flag_no_login_60days'] = (df['days_inactive'] >= 60).astype(int)
    df['flag_low_feature_adoption'] = (df['feature_adoption_rate'] < 0.30).astype(int)
    df['flag_high_support_tickets'] = (df['support_tickets'] >= 3).astype(int)
    df['flag_declining_engagement'] = (df['logins_trend_3m'] < -0.10).astype(int)
    
    # Risk flag count
    risk_flags = ['flag_no_login_30days', 'flag_low_feature_adoption', 'flag_high_support_tickets', 'flag_declining_engagement']
    df['risk_flag_count'] = df[risk_flags].sum(axis=1)
    
    # Health score
    df['health_score'] = (
        (df['recency_score'] * 25) +
        (df['feature_adoption_rate'] * 25) +
        ((100 - df['engagement_decay'].clip(-100, 100)) / 100 * 25) +
        (1 - df['support_tickets'] / 5 * 25)
    ).clip(0, 100)
    
    return df


def score_customers(db_path='data/churn_system.db', model_path='models/churn_model_v1.pkl'):
    """
    Score all customers with trained model.
    """
    
    print("\n" + "=" * 70)
    print("CUSTOMER SCORING")
    print("=" * 70)
    
    # Load data from database
    print("\nLoading data from database...")
    conn = sqlite3.connect(db_path)
    
    query = '''
    SELECT 
        e.*,
        c.monthly_revenue,
        c.subscription_tier,
        c.company_size
    FROM engagement_raw e
    JOIN customers c ON e.customer_id = c.customer_id
    ORDER BY e.customer_id, e.year_month
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"✓ Loaded {len(df):,} records for {df['customer_id'].nunique():,} customers")
    
    # Engineer features
    print("\nEngineering features...")
    df = engineer_features_for_prediction(df)
    print("✓ Features engineered")
    
    # Load model
    print("\nLoading trained model...")
    model = joblib.load(model_path)
    print("✓ Model loaded")
    
    # Get latest month only
    print("\nPreparing for scoring...")
    latest_month = df['year_month'].max()
    latest_df = df[df['year_month'] == latest_month].copy()
    
    # Select features that model expects
    feature_cols = [
        'logins_per_day', 'api_calls', 'features_used_count', 'days_inactive',
        'support_tickets', 'engagement_score', 'logins_3m_avg', 'api_calls_3m_avg',
        'features_3m_avg', 'logins_trend_3m', 'engagement_decay', 'recency_score',
        'feature_adoption_rate', 'is_high_adoption', 'is_low_adoption',
        'flag_no_login_30days', 'flag_no_login_60days', 'flag_low_feature_adoption',
        'flag_high_support_tickets', 'flag_declining_engagement', 'risk_flag_count',
        'health_score'
    ]
    
    X = latest_df[feature_cols].fillna(0)
    
    # Score
    print("Scoring with model...")
    churn_probs = model.predict_proba(X)[:, 1]
    
    # Prepare results
    results = latest_df[['customer_id', 'monthly_revenue']].copy()
    results['prediction_date'] = latest_month
    results['churn_probability'] = churn_probs
    results['predicted_churn'] = model.predict(X)
    results['health_score'] = latest_df['health_score']
    
    # Risk tier
    results['risk_tier'] = pd.cut(
        churn_probs,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # Revenue at risk
    results['revenue_at_risk'] = results['monthly_revenue'] * churn_probs * 3
    results['annual_revenue_at_risk'] = results['monthly_revenue'] * churn_probs * 12
    
    # Risk score
    results['composite_risk_score'] = churn_probs
    results['risk_rank'] = results['composite_risk_score'].rank(method='first', ascending=False).astype(int)
    
    # Recommendation
    def assign_rec(prob, health):
        if prob > 0.6 and health < 40:
            return 'Urgent Account Review'
        elif prob > 0.6 and health < 50:
            return 'Onboarding Support'
        elif prob > 0.5:
            return 'Engagement Campaign'
        elif prob > 0.3:
            return 'Executive Check-in'
        else:
            return 'Monitor'
    
    results['recommendation'] = results.apply(
        lambda x: assign_rec(x['churn_probability'], x['health_score']),
        axis=1
    )
    
    print(f"✓ Scored {len(results):,} customers")
    print(f"\nPrediction Summary:")
    print(f"  High Risk: {(results['risk_tier'] == 'High Risk').sum():,}")
    print(f"  Medium Risk: {(results['risk_tier'] == 'Medium Risk').sum():,}")
    print(f"  Low Risk: {(results['risk_tier'] == 'Low Risk').sum():,}")
    print(f"  Revenue at Risk: ${results['revenue_at_risk'].sum():,.0f}")
    
    # Store in database
    store_predictions(results, db_path)
    
    return results


def store_predictions(predictions_df, db_path='data/churn_system.db'):
    """Store predictions in database."""
    
    print("\nStoring predictions in database...")
    
    conn = sqlite3.connect(db_path)
    
    try:
        predictions_df.to_sql('predictions', conn, if_exists='append', index=False)
        conn.commit()
        print(f"✓ Stored {len(predictions_df):,} predictions")
    except Exception as e:
        conn.rollback()
        print(f"✗ Error storing predictions: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    score_customers()