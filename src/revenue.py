# src/revenue.py
"""
Revenue At-Risk Calculation & Recommendations

Translates churn predictions into business impact.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

def generate_predictions(features_df):
    """
    Generate churn predictions using trained model.
    """
    print("\nGenerating churn predictions...")
    
    # Load model
    model = joblib.load('models/churn_model_v1.pkl')
    
    # Prepare features
    exclude_cols = ['customer_id', 'year_month', 'churned', 
                    'company_size', 'subscription_tier', 'industry', 'monthly_revenue']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_cols].fillna(features_df[feature_cols].mean())
    
    # Get predictions
    churn_probs = model.predict_proba(X)[:, 1]
    
    # Add to dataframe
    features_df['churn_probability'] = churn_probs
    features_df['predicted_churn'] = model.predict(X)
    features_df['risk_tier'] = pd.cut(churn_probs, 
                                       bins=[0, 0.3, 0.6, 1.0],
                                       labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    print(f"✓ Predictions generated")
    return features_df


def calculate_revenue_at_risk(features_df, customers_df):
    """
    Calculate revenue at risk for each customer.
    
    Formula:
    Revenue At Risk = Monthly Revenue × Churn Probability × Retention Period
    """
    
    print("\n" + "=" * 70)
    print("REVENUE AT-RISK CALCULATION")
    print("=" * 70)
    
    # Merge predictions with customer revenue data
    df = features_df[['customer_id', 'churn_probability', 'risk_tier', 'health_score']].copy()
    df = df.merge(customers_df[['customer_id', 'monthly_revenue', 'subscription_tier']], 
                  on='customer_id', how='left')
    
    # Calculate revenue at risk
    # Assumption: retention period = 3 months (typical notice period)
    retention_period_months = 3
    
    df['revenue_at_risk'] = (
        df['monthly_revenue'] * 
        df['churn_probability'] * 
        retention_period_months
    )
    
    # Annual projection (if customer churns)
    df['annual_revenue_at_risk'] = df['monthly_revenue'] * df['churn_probability'] * 12
    
    print(f"\n✓ Calculated revenue at risk for {len(df):,} customers")
    print(f"\nRevenue At Risk Statistics:")
    print(f"  Total 3-month at risk: ${df['revenue_at_risk'].sum():,.0f}")
    print(f"  Total annual at risk: ${df['annual_revenue_at_risk'].sum():,.0f}")
    print(f"  Average per customer: ${df['revenue_at_risk'].mean():,.0f}")
    print(f"  Median per customer: ${df['revenue_at_risk'].median():,.0f}")
    
    return df


def calculate_risk_scores(df):
    """
    Create composite risk scores.
    """
    
    print("\n" + "=" * 70)
    print("RISK SCORING")
    print("=" * 70)
    
    # Normalize metrics (0-1 scale)
    df['churn_prob_norm'] = df['churn_probability']
    df['health_norm'] = 1 - (df['health_score'] / 100)  # Inverse
    df['revenue_norm'] = (df['monthly_revenue'] - df['monthly_revenue'].min()) / (df['monthly_revenue'].max() - df['monthly_revenue'].min())
    
    # Composite risk score
    df['composite_risk_score'] = (
        df['churn_prob_norm'] * 0.50 +
        df['health_norm'] * 0.30 +
        df['revenue_norm'] * 0.20
    )
    
    # Risk rank
    df['risk_rank'] = df['composite_risk_score'].rank(method='first', ascending=False).astype(int)
    
    print(f"\n✓ Composite risk scores calculated")
    
    return df


def create_recommendations(df):
    """
    Create retention recommendations based on risk profile.
    """
    
    print("\n" + "=" * 70)
    print("RETENTION RECOMMENDATIONS")
    print("=" * 70)
    
    def assign_recommendation(row):
        churn_prob = row['churn_probability']
        health = row['health_score']
        
        if churn_prob > 0.6 and health < 40:
            return 'Urgent Account Review'
        elif churn_prob > 0.6 and health < 50:
            return 'Onboarding Support'
        elif churn_prob > 0.5:
            return 'Engagement Campaign'
        elif churn_prob > 0.3:
            return 'Executive Check-in'
        else:
            return 'Monitor'
    
    df['recommendation'] = df.apply(assign_recommendation, axis=1)
    
    print(f"\n✓ Recommendations assigned")
    print(f"\nRecommendations Distribution:")
    print(df['recommendation'].value_counts())
    
    return df


def create_priority_list(df, top_n=50):
    """
    Create prioritized list of customers for retention efforts.
    """
    
    print("\n" + "=" * 70)
    print(f"TOP {top_n} PRIORITY CUSTOMERS")
    print("=" * 70)
    
    priority = df.nlargest(top_n, 'revenue_at_risk')[
        ['customer_id', 'monthly_revenue', 'churn_probability', 'health_score', 
         'revenue_at_risk', 'recommendation', 'risk_rank']
    ].copy()
    
    priority = priority.reset_index(drop=True)
    priority['priority_rank'] = range(1, len(priority) + 1)
    
    print(f"\nTop 10:")
    print(priority.head(10)[['priority_rank', 'customer_id', 'monthly_revenue', 
                             'churn_probability', 'recommendation']].to_string(index=False))
    
    return priority


def calculate_business_metrics(df):
    """
    Calculate key business metrics for reporting.
    """
    
    print("\n" + "=" * 70)
    print("BUSINESS METRICS")
    print("=" * 70)
    
    metrics = {
        'total_customers': len(df),
        'total_mrr': df['monthly_revenue'].sum(),
        'high_risk_count': (df['risk_tier'] == 'High Risk').sum(),
        'high_risk_pct': (df['risk_tier'] == 'High Risk').mean() * 100,
        'revenue_at_risk_3m': df['revenue_at_risk'].sum(),
        'revenue_at_risk_annual': df['annual_revenue_at_risk'].sum(),
        'avg_churn_prob': df['churn_probability'].mean(),
        'urgent_customers': (df['recommendation'] == 'Urgent Account Review').sum(),
    }
    
    print(f"\nTotal Customers: {metrics['total_customers']:,}")
    print(f"Total Monthly Revenue: ${metrics['total_mrr']:,.0f}")
    print(f"\nHigh Risk Customers: {metrics['high_risk_count']:,} ({metrics['high_risk_pct']:.1f}%)")
    print(f"Revenue At Risk (3 months): ${metrics['revenue_at_risk_3m']:,.0f}")
    print(f"Revenue At Risk (12 months): ${metrics['revenue_at_risk_annual']:,.0f}")
    print(f"\nAverage Churn Probability: {metrics['avg_churn_prob']:.1%}")
    print(f"Customers Needing Urgent Review: {metrics['urgent_customers']:,}")
    
    return metrics


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PHASE 4: REVENUE AT-RISK & BUSINESS LOGIC")
    print("=" * 70)
    
    # Load data
    features = pd.read_csv('data/processed/features.csv')
    customers = pd.read_csv('data/raw/customers.csv')
    
    # Step 0: Generate predictions
    features = generate_predictions(features)
    
    # Step 1: Calculate revenue at risk
    revenue_df = calculate_revenue_at_risk(features, customers)
    
    # Step 2: Calculate risk scores
    revenue_df = calculate_risk_scores(revenue_df)
    
    # Step 3: Create recommendations
    revenue_df = create_recommendations(revenue_df)
    
    # Step 4: Create priority list
    priority_list = create_priority_list(revenue_df, top_n=50)
    
    # Step 5: Calculate business metrics
    metrics = calculate_business_metrics(revenue_df)
    
    # Save results
    revenue_df.to_csv('data/processed/revenue_at_risk.csv', index=False)
    priority_list.to_csv('data/processed/priority_customers.csv', index=False)
    
    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)
    print(f"\n✓ Revenue at risk: data/processed/revenue_at_risk.csv")
    print(f"✓ Priority list: data/processed/priority_customers.csv")