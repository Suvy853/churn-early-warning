# src/prescriptions.py
"""
Prescriptive AI: Generate Recommended Actions with ROI

Not just predicting churn, but recommending what action to take
and what the expected ROI is.
"""

import pandas as pd
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)


def generate_prescriptions(db_path='data/churn_system.db'):
    """
    For each customer, recommend an intervention with ROI analysis.
    
    Returns:
        DataFrame with prescriptions for all customers
    """
    
    print("\n" + "=" * 70)
    print("GENERATING PRESCRIPTIONS (RECOMMENDED ACTIONS)")
    print("=" * 70)
    
    # Load latest predictions from database
    conn = sqlite3.connect(db_path)
    
    query = '''
    SELECT * FROM predictions
    WHERE prediction_date = (SELECT MAX(prediction_date) FROM predictions)
    '''
    
    predictions = pd.read_sql_query(query, conn)
    conn.close()
    
    if predictions.empty:
        print("✗ No predictions found in database. Run phase 5 first.")
        return pd.DataFrame()
    
    prescriptions = []
    
    for _, customer in predictions.iterrows():
        customer_id = customer['customer_id']
        churn_prob = customer['churn_probability']
        revenue = customer['monthly_revenue']
        health_score = customer['health_score']
        recommendation = customer['recommendation']
        
        # Generate prescription based on risk profile
        if recommendation == 'Urgent Account Review' and churn_prob > 0.6 and health_score < 40:
            # Critical: High churn + low health → Aggressive intervention
            prescription = {
                'customer_id': customer_id,
                'churn_probability': churn_prob,
                'current_recommendation': recommendation,
                'prescribed_action': 'Urgent Discount + Personal Outreach',
                'action_type': 'discount_and_outreach',
                'discount_percent': 20,
                'action_cost': revenue * 0.20,  # 20% of monthly revenue
                'expected_retention_value': revenue * 12,  # 1 year of revenue if saved
                'expected_impact': revenue * 12 * (1 - churn_prob),
                'roi': (revenue * 12 * (1 - churn_prob)) / (revenue * 0.20) if revenue > 0 else 0,
                'confidence': 0.85,
                'timing': 'Within 24 hours',
                'rationale': 'Critical churn risk. Immediate intervention needed.',
                'success_probability': 0.50
            }
        
        elif recommendation == 'Onboarding Support' and churn_prob > 0.5:
            # High churn + training needed → Feature adoption focus
            prescription = {
                'customer_id': customer_id,
                'churn_probability': churn_prob,
                'current_recommendation': recommendation,
                'prescribed_action': 'Guided Feature Adoption Program',
                'action_type': 'training',
                'discount_percent': 10,
                'action_cost': 500,  # Fixed cost for training
                'expected_retention_value': revenue * 12,
                'expected_impact': revenue * 12 * (1 - churn_prob),
                'roi': (revenue * 12 * (1 - churn_prob)) / 500 if revenue > 0 else 0,
                'confidence': 0.72,
                'timing': 'This week',
                'rationale': 'Customer not using features. Training can unlock value.',
                'success_probability': 0.45
            }
        
        elif recommendation == 'Engagement Campaign' and churn_prob > 0.4:
            # Medium churn → Re-engagement
            prescription = {
                'customer_id': customer_id,
                'churn_probability': churn_prob,
                'current_recommendation': recommendation,
                'prescribed_action': 'Multi-Channel Re-engagement Campaign',
                'action_type': 'engagement',
                'discount_percent': 15,
                'action_cost': 750,  # Campaign cost
                'expected_retention_value': revenue * 12,
                'expected_impact': revenue * 12 * (1 - churn_prob),
                'roi': (revenue * 12 * (1 - churn_prob)) / 750 if revenue > 0 else 0,
                'confidence': 0.68,
                'timing': 'Next 2 weeks',
                'rationale': 'Engagement declining. Multi-touch campaign to re-activate.',
                'success_probability': 0.40
            }
        
        elif recommendation == 'Executive Check-in' and churn_prob > 0.3 and revenue > 3000:
            # Medium churn + high value → Executive touch
            prescription = {
                'customer_id': customer_id,
                'churn_probability': churn_prob,
                'current_recommendation': recommendation,
                'prescribed_action': 'Executive Account Review',
                'action_type': 'executive_outreach',
                'discount_percent': 12,
                'action_cost': 2000,  # Executive time cost
                'expected_retention_value': revenue * 24,  # 2-year value
                'expected_impact': revenue * 24 * (1 - churn_prob),
                'roi': (revenue * 24 * (1 - churn_prob)) / 2000 if revenue > 0 else 0,
                'confidence': 0.75,
                'timing': 'Next 3 weeks',
                'rationale': 'High-value customer. Executive relationship management.',
                'success_probability': 0.55
            }
        
        else:
            # Low risk → Monitor only
            prescription = {
                'customer_id': customer_id,
                'churn_probability': churn_prob,
                'current_recommendation': recommendation,
                'prescribed_action': 'Monitor and Nurture',
                'action_type': 'monitor',
                'discount_percent': 0,
                'action_cost': 0,
                'expected_retention_value': 0,
                'expected_impact': 0,
                'roi': 0,
                'confidence': 0.95,
                'timing': 'Ongoing',
                'rationale': 'Customer is healthy. Standard engagement practices.',
                'success_probability': 0.90
            }
        
        prescriptions.append(prescription)
    
    prescriptions_df = pd.DataFrame(prescriptions)
    
    # Sort by ROI (descending) - highest value opportunities first
    prescriptions_df = prescriptions_df.sort_values('roi', ascending=False)
    
    print(f"\n✓ Generated prescriptions for {len(prescriptions_df):,} customers")
    
    print(f"\nTop 10 Highest ROI Actions:")
    print(prescriptions_df.head(10)[
        ['customer_id', 'prescribed_action', 'roi', 'confidence', 'success_probability']
    ].to_string(index=False))
    
    # Calculate portfolio-level impact
    total_action_cost = prescriptions_df['action_cost'].sum()
    total_expected_impact = prescriptions_df['expected_impact'].sum()
    portfolio_roi = total_expected_impact / total_action_cost if total_action_cost > 0 else 0
    
    print(f"\n" + "=" * 70)
    print("PORTFOLIO-LEVEL IMPACT")
    print("=" * 70)
    print(f"Total intervention cost: ${total_action_cost:,.0f}")
    print(f"Total expected impact: ${total_expected_impact:,.0f}")
    print(f"Portfolio ROI: {portfolio_roi:.1f}x")
    
    # Breakdown by action type
    print(f"\nBreakdown by Action Type:")
    action_breakdown = prescriptions_df.groupby('action_type').agg({
        'customer_id': 'count',
        'action_cost': 'sum',
        'expected_impact': 'sum',
        'roi': 'mean'
    }).round(0)
    action_breakdown.columns = ['Customer Count', 'Total Cost', 'Total Impact', 'Avg ROI']
    print(action_breakdown)
    
    return prescriptions_df


def save_prescriptions(prescriptions_df, db_path='data/churn_system.db'):
    """Save prescriptions to database."""
    
    print(f"\nSaving prescriptions to database...")
    
    conn = sqlite3.connect(db_path)
    
    try:
        prescriptions_df.to_sql('prescriptions', conn, if_exists='replace', index=False)
        conn.commit()
        print(f"✓ Saved {len(prescriptions_df):,} prescriptions")
    except Exception as e:
        conn.rollback()
        print(f"✗ Error saving prescriptions: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    # Generate prescriptions
    prescriptions = generate_prescriptions()
    
    # Save to database
    if not prescriptions.empty:
        save_prescriptions(prescriptions)
        
        print(f"\n✓ Prescriptions complete!")
        print(f"  File: data/prescriptions.csv")