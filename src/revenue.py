# src/revenue.py
"""
Revenue Impact Calculation

Calculates revenue at-risk and classifies customers into risk tiers.
"""

import pandas as pd
import sqlite3


def get_db_connection(db_path='data/churn_system.db'):
    """Get database connection."""
    conn = sqlite3.connect(db_path)
    return conn


def calculate_risk_tier(churn_probability):
    """
    Classify customer into risk tier based on churn probability.
    
    Updated thresholds:
    - Low Risk: < 20%
    - Medium Risk: 20-40%
    - High Risk: > 40%
    """
    if churn_probability < 0.20:
        return 'Low Risk'
    elif churn_probability < 0.40:
        return 'Medium Risk'
    else:
        return 'High Risk'


def calculate_revenue_at_risk(monthly_revenue, churn_probability, months=3):
    """
    Calculate revenue at-risk.
    
    Formula: Monthly Revenue × Churn Probability × Number of Months
    
    Example: $2500 × 0.78 × 3 = $5,850 annual revenue at-risk
    """
    return monthly_revenue * churn_probability * months


def add_revenue_metrics(predictions_df, customers_df):
    """
    Add revenue and risk tier calculations to predictions.
    
    Merges customer revenue data with predictions and calculates:
    - Monthly revenue
    - Annual revenue at-risk
    - Risk tier
    - Risk rank (rank customers by revenue at-risk)
    """
    
    # Merge predictions with customer revenue data
    merged = predictions_df.merge(
        customers_df[['customer_id', 'monthly_revenue']],
        on='customer_id',
        how='left'
    )
    
    # Calculate revenue at-risk
    merged['revenue_at_risk'] = merged.apply(
        lambda row: calculate_revenue_at_risk(
            row['monthly_revenue'],
            row['churn_probability']
        ),
        axis=1
    )
    
    # Calculate annual revenue at-risk (multiply by 4 quarters)
    merged['annual_revenue_at_risk'] = merged['revenue_at_risk'] * 4
    
    # Classify into risk tier
    merged['risk_tier'] = merged['churn_probability'].apply(calculate_risk_tier)
    
    # Rank customers by revenue at-risk (highest risk first)
    merged['risk_rank'] = merged['annual_revenue_at_risk'].rank(ascending=False, method='min').astype(int)
    
    return merged


def get_revenue_summary(predictions_with_revenue):
    """
    Get summary statistics on revenue impact.
    """
    
    summary = {
        'total_customers': len(predictions_with_revenue),
        'low_risk_count': len(predictions_with_revenue[predictions_with_revenue['risk_tier'] == 'Low Risk']),
        'medium_risk_count': len(predictions_with_revenue[predictions_with_revenue['risk_tier'] == 'Medium Risk']),
        'high_risk_count': len(predictions_with_revenue[predictions_with_revenue['risk_tier'] == 'High Risk']),
        'total_annual_revenue': predictions_with_revenue['monthly_revenue'].sum() * 12,
        'total_revenue_at_risk': predictions_with_revenue['annual_revenue_at_risk'].sum(),
        'avg_churn_probability': predictions_with_revenue['churn_probability'].mean(),
        'high_risk_revenue_at_risk': predictions_with_revenue[
            predictions_with_revenue['risk_tier'] == 'High Risk'
        ]['annual_revenue_at_risk'].sum(),
    }
    
    return summary


def save_revenue_metrics(predictions_with_revenue, output_path='data/processed/revenue_at_risk.csv'):
    """Save revenue metrics to CSV for later analysis."""
    
    predictions_with_revenue.to_csv(output_path, index=False)
    print(f"✓ Revenue metrics saved to {output_path}")


def get_priority_customers(predictions_with_revenue, limit=50):
    """
    Get top N customers by revenue at-risk.
    
    These are the customers to focus retention efforts on.
    """
    
    priority = predictions_with_revenue.nlargest(
        limit,
        'annual_revenue_at_risk'
    )[['customer_id', 'monthly_revenue', 'churn_probability', 
       'health_score', 'revenue_at_risk', 'annual_revenue_at_risk', 
       'risk_tier', 'recommendation']]
    
    return priority


if __name__ == "__main__":
    # Example usage
    print("Revenue Impact Calculation Module")
    print("=" * 60)
    print("This module calculates revenue at-risk for each customer.")
    print("\nRisk Tier Thresholds:")
    print("  Low Risk: < 20% churn probability")
    print("  Medium Risk: 20-40% churn probability")
    print("  High Risk: > 40% churn probability")
    print("\nFormula: Revenue at-Risk = Monthly Revenue × Churn Prob × 3 months")