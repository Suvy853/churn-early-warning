# src/features.py
"""
Feature Engineering for Churn Prediction

Transforms raw engagement metrics into ML-ready features.
Each feature encodes business logic that predicts churn.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

def create_features(engagement_df, customers_df):
    """
    Main feature engineering function.
    
    Takes raw engagement data and creates predictive features.
    
    Args:
        engagement_df: Monthly engagement data (from Phase 1)
        customers_df: Customer attributes
    
    Returns:
        features_df: Engineered features ready for ML
    """
    
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)
    
    # Step 1: Merge engagement with customer data
    print("\nStep 1: Merging data...")
    df = engagement_df.merge(customers_df, on='customer_id')
    
    # Step 2: Sort by customer and time (required for rolling calculations)
    print("Step 2: Sorting by customer and time...")
    df = df.sort_values(['customer_id', 'year_month']).reset_index(drop=True)
    
    # Step 3: Create rolling average features (7-month, 3-month, 1-month windows)
    print("Step 3: Calculating rolling averages...")
    df = create_rolling_features(df)
    
    # Step 4: Create trend features (is engagement increasing or decreasing?)
    print("Step 4: Creating trend features...")
    df = create_trend_features(df)
    
    # Step 5: Create recency features (how recently active?)
    print("Step 5: Creating recency features...")
    df = create_recency_features(df)
    
    # Step 6: Create adoption features (feature usage patterns)
    print("Step 6: Creating adoption features...")
    df = create_adoption_features(df)
    
    # Step 7: Create risk flag features (binary risk indicators)
    print("Step 7: Creating risk flags...")
    df = create_risk_flags(df)
    
    # Step 8: Create composite features (combine multiple signals)
    print("Step 8: Creating composite features...")
    df = create_composite_features(df)
    
    # Step 9: Select final features for ML
    print("Step 9: Selecting final features...")
    feature_cols = select_final_features(df)
    features_df = df[feature_cols].copy()
    
    print(f"\n✓ Created {len(feature_cols)} features")
    print(f"✓ Feature matrix shape: {features_df.shape}")
    
    return features_df


def create_rolling_features(df):
    """
    Create rolling average features.
    
    Rolling averages smooth out noise and show trends over time.
    """
    
    # For each customer, calculate rolling averages
    for customer_id in df['customer_id'].unique():
        mask = df['customer_id'] == customer_id
        customer_data = df[mask].copy()
        
        # 3-month rolling average of logins
        df.loc[mask, 'logins_3m_avg'] = customer_data['logins_per_day'].rolling(3, min_periods=1).mean()
        
        # 3-month rolling average of API calls
        df.loc[mask, 'api_calls_3m_avg'] = customer_data['api_calls'].rolling(3, min_periods=1).mean()
        
        # 3-month rolling average of features used
        df.loc[mask, 'features_3m_avg'] = customer_data['features_used_count'].rolling(3, min_periods=1).mean()
    
    # Fill any NaNs (first month won't have 3-month history)
    df['logins_3m_avg'] = df['logins_3m_avg'].fillna(df['logins_per_day'])
    df['api_calls_3m_avg'] = df['api_calls_3m_avg'].fillna(df['api_calls'])
    df['features_3m_avg'] = df['features_3m_avg'].fillna(df['features_used_count'])
    
    return df


def create_trend_features(df):
    """
    Create trend features (is engagement going up or down?).
    
    Key insight: Declining engagement predicts churn.
    """
    
    # For each customer, calculate trend (slope of engagement over time)
    for customer_id in df['customer_id'].unique():
        mask = df['customer_id'] == customer_id
        customer_data = df[mask].copy()
        
        # Trend: compare last 3 months vs previous 3 months
        recent_logins = customer_data['logins_per_day'].tail(3).mean()
        previous_logins = customer_data['logins_per_day'].iloc[-6:-3].mean() if len(customer_data) >= 6 else customer_data['logins_per_day'].iloc[0]
        
        # Calculate change (positive = improving, negative = declining)
        if previous_logins > 0:
            login_trend = (recent_logins - previous_logins) / previous_logins
        else:
            login_trend = 0
        
        # Assign trend to all rows for this customer (for this month)
        df.loc[mask, 'logins_trend_3m'] = login_trend
    
    # Engagement decay: how fast is engagement dropping?
    df['engagement_decay'] = df.groupby('customer_id')['engagement_score'].diff()
    df['engagement_decay'] = df['engagement_decay'].fillna(0)
    
    return df


def create_recency_features(df):
    """
    Create recency features (how recently was customer active?).
    
    Recency is a strong predictor of churn.
    """
    
    # Days inactive (already in raw data)
    df['days_inactive'] = df['days_since_last_login']
    
    # Categorical recency (binned)
    df['recency_category'] = pd.cut(
        df['days_inactive'],
        bins=[0, 7, 14, 30, 60, 365],
        labels=['0-7 days', '7-14 days', '14-30 days', '30-60 days', '60+ days']
    )
    
    # Recency score (0-1, where 1 = recently active)
    df['recency_score'] = 1 - (df['days_inactive'] / 60).clip(0, 1)
    
    return df


def create_adoption_features(df):
    """
    Create adoption features (what % of features are being used?).
    """
    
    # Feature adoption rate (0-1)
    df['feature_adoption_rate'] = df['features_used_count'] / 10.0
    
    # High adoption (uses >70% of features)
    df['is_high_adoption'] = (df['feature_adoption_rate'] > 0.70).astype(int)
    
    # Low adoption (uses <30% of features)
    df['is_low_adoption'] = (df['feature_adoption_rate'] < 0.30).astype(int)
    
    return df


def create_risk_flags(df):
    """
    Create binary risk flag features (yes/no indicators).
    
    These are highly interpretable and useful for business teams.
    """
    
    # Flag: User hasn't logged in for 30+ days (very concerning)
    df['flag_no_login_30days'] = (df['days_inactive'] >= 30).astype(int)
    
    # Flag: User hasn't logged in for 60+ days (critical)
    df['flag_no_login_60days'] = (df['days_inactive'] >= 60).astype(int)
    
    # Flag: Low feature adoption (not using the product)
    df['flag_low_feature_adoption'] = (df['feature_adoption_rate'] < 0.30).astype(int)
    
    # Flag: High support tickets (sign of problems)
    df['flag_high_support_tickets'] = (df['support_tickets'] >= 3).astype(int)
    
    # Flag: Declining engagement (trending down)
    df['flag_declining_engagement'] = (df['logins_trend_3m'] < -0.10).astype(int)
    
    return df


def create_composite_features(df):
    """
    Create composite features (combine multiple signals).
    
    Sometimes multiple weak signals are stronger than one strong signal.
    """
    
    # Risk score: how many risk flags are present?
    risk_flags = [
        'flag_no_login_30days',
        'flag_low_feature_adoption',
        'flag_high_support_tickets',
        'flag_declining_engagement'
    ]
    df['risk_flag_count'] = df[risk_flags].sum(axis=1)
    
    # Health score (0-100)
    # Combines: recency, adoption, engagement, support
    df['health_score'] = (
        (df['recency_score'] * 25) +  # Recent activity = healthy
        (df['feature_adoption_rate'] * 25) +  # Feature usage = healthy
        ((100 - df['engagement_decay'].clip(-100, 100)) / 100 * 25) +  # Stable engagement = healthy
        (1 - df['support_tickets'] / 5 * 25)  # Low support = healthy
    ).clip(0, 100)
    
    return df


def select_final_features(df):
    """
    Select which features to use for ML model.
    
    Strategy: Include interpretable features that have business meaning.
    """
    
    final_features = [
        # Identity
        'customer_id',
        'year_month',
        'churned',  # Target variable
        
        # Raw metrics
        'logins_per_day',
        'api_calls',
        'features_used_count',
        'days_inactive',
        'support_tickets',
        'engagement_score',
        
        # Rolling averages
        'logins_3m_avg',
        'api_calls_3m_avg',
        'features_3m_avg',
        
        # Trend features
        'logins_trend_3m',
        'engagement_decay',
        
        # Recency features
        'recency_score',
        
        # Adoption features
        'feature_adoption_rate',
        'is_high_adoption',
        'is_low_adoption',
        
        # Risk flags
        'flag_no_login_30days',
        'flag_no_login_60days',
        'flag_low_feature_adoption',
        'flag_high_support_tickets',
        'flag_declining_engagement',
        'risk_flag_count',
        
        # Composite features
        'health_score',
        
        # Customer segment (for analysis)
        'company_size',
        'subscription_tier',
        'industry',
        'monthly_revenue',
    ]
    
    return final_features


if __name__ == "__main__":
    # Test feature engineering
    customers = pd.read_csv('data/raw/customers.csv')
    engagement = pd.read_csv('data/raw/engagement_monthly.csv')
    
    features = create_features(engagement, customers)
    
    # Save features
    features.to_csv('data/processed/features.csv', index=False)
    print(f"\n✓ Features saved to: data/processed/features.csv")