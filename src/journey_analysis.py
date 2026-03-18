# src/journey_analysis.py
"""
Customer Journey Analysis

Tracks the customer's engagement journey over time.
Identifies inflection points and critical moments.
Shows when customer became at-risk.
"""

import pandas as pd
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)


def analyze_customer_journey(customer_id, db_path='data/churn_system.db'):
    """
    Analyze a specific customer's engagement journey.
    
    Returns:
        Dictionary with journey data and analysis
    """
    
    conn = sqlite3.connect(db_path)
    
    query = '''
    SELECT * FROM engagement_raw
    WHERE customer_id = ?
    ORDER BY year_month ASC
    '''
    
    journey = pd.read_sql_query(query, conn, params=(customer_id,))
    conn.close()
    
    if journey.empty:
        return None
    
    # Analyze phases and trends
    journey_analysis = []
    
    for idx, row in journey.iterrows():
        month = row['year_month']
        logins = row['logins_per_day']
        features = row['features_used_count']
        engagement = row['engagement_score']
        days_inactive = row['days_since_last_login']
        support_tickets = row['support_tickets']
        churned = row['churned']
        
        # Detect health phase
        if logins < 0.5 and days_inactive > 30:
            phase = 'Disengaged'
            health_indicator = 'Critical'
        elif logins < 1.0 and engagement < 40:
            phase = 'Declining'
            health_indicator = 'Warning'
        elif features < 3:
            phase = 'Low Adoption'
            health_indicator = 'At Risk'
        elif engagement < 50:
            phase = 'Moderate'
            health_indicator = 'Monitor'
        else:
            phase = 'Healthy'
            health_indicator = 'Good'
        
        # Churn risk indicator
        if churned == 1:
            risk = 'CHURNED'
        elif logins < 0.5:
            risk = 'Critical'
        elif logins < 1.0 or features < 3:
            risk = 'High'
        elif engagement < 50:
            risk = 'Medium'
        else:
            risk = 'Low'
        
        journey_analysis.append({
            'month': month,
            'logins_per_day': logins,
            'features_used': features,
            'engagement_score': engagement,
            'days_inactive': days_inactive,
            'support_tickets': support_tickets,
            'phase': phase,
            'health': health_indicator,
            'risk': risk,
            'churned': churned
        })
    
    journey_df = pd.DataFrame(journey_analysis)
    
    # Find inflection points (when phase changed)
    journey_df['phase_changed'] = journey_df['phase'].ne(journey_df['phase'].shift()).fillna(False).astype(int)
    inflection_points = journey_df[journey_df['phase_changed'] == 1]
    
    # Find when customer became at-risk
    at_risk_rows = journey_df[journey_df['risk'].isin(['Critical', 'High'])]
    first_at_risk = at_risk_rows.iloc[0] if len(at_risk_rows) > 0 else None
    
    return {
        'customer_id': customer_id,
        'journey': journey_df,
        'inflection_points': inflection_points,
        'current_phase': journey_df.iloc[-1]['phase'],
        'current_risk': journey_df.iloc[-1]['risk'],
        'current_health': journey_df.iloc[-1]['health'],
        'churned': journey_df.iloc[-1]['churned'] == 1,
        'first_at_risk_month': first_at_risk['month'] if first_at_risk is not None else None,
        'months_until_churn': len(journey_df) - at_risk_rows.index[0] if len(at_risk_rows) > 0 else None
    }


def generate_all_journeys(db_path='data/churn_system.db'):
    """
    Generate journey analysis for all customers.
    
    Returns:
        DataFrame with summary for all customers
    """
    
    print("\n" + "=" * 70)
    print("CUSTOMER JOURNEY ANALYSIS")
    print("=" * 70)
    
    conn = sqlite3.connect(db_path)
    customer_ids = pd.read_sql_query(
        'SELECT DISTINCT customer_id FROM engagement_raw ORDER BY customer_id',
        conn
    )
    conn.close()
    
    all_journeys = []
    
    print(f"\nAnalyzing {len(customer_ids)} customers...")
    
    for idx, row in customer_ids.iterrows():
        customer_id = row['customer_id']
        journey = analyze_customer_journey(customer_id, db_path)
        
        if journey:
            all_journeys.append({
                'customer_id': customer_id,
                'current_phase': journey['current_phase'],
                'current_risk': journey['current_risk'],
                'current_health': journey['current_health'],
                'churned': journey['churned'],
                'first_at_risk_month': journey['first_at_risk_month'],
                'months_until_churn': journey['months_until_churn'],
                'total_months_observed': len(journey['journey'])
            })
    
    summary_df = pd.DataFrame(all_journeys)
    
    print(f"\n✓ Journey analysis complete")
    
    # Statistics
    print(f"\nJourney Statistics:")
    print(f"  Total customers analyzed: {len(summary_df):,}")
    print(f"  Churned customers: {summary_df['churned'].sum():,} ({summary_df['churned'].mean()*100:.1f}%)")
    print(f"  Currently at-risk: {(summary_df['current_risk'].isin(['Critical', 'High'])).sum():,}")
    
    # Phase distribution
    print(f"\nCurrent Phase Distribution:")
    phase_dist = summary_df['current_phase'].value_counts()
    for phase, count in phase_dist.items():
        print(f"  {phase}: {count:,} ({count/len(summary_df)*100:.1f}%)")
    
    # Risk distribution
    print(f"\nCurrent Risk Distribution:")
    risk_dist = summary_df['current_risk'].value_counts()
    for risk, count in risk_dist.items():
        print(f"  {risk}: {count:,} ({count/len(summary_df)*100:.1f}%)")
    
    return summary_df


def get_journey_narrative(customer_id, db_path='data/churn_system.db'):
    """
    Generate a human-readable narrative of customer's journey.
    """
    
    journey = analyze_customer_journey(customer_id, db_path)
    
    if not journey:
        return f"No data for customer {customer_id}"
    
    narrative = f"""
CUSTOMER {customer_id} JOURNEY ANALYSIS
{'='*60}

Current Status:
  Phase: {journey['current_phase']}
  Health: {journey['current_health']}
  Risk Level: {journey['current_risk']}
  Churned: {'Yes' if journey['churned'] else 'No'}

Journey Timeline:
  Total months observed: {len(journey['journey'])}
  First at-risk: {journey['first_at_risk_month'] if journey['first_at_risk_month'] else 'Never'}
  {f"Months until churn: {journey['months_until_churn']}" if journey['months_until_churn'] else ""}

Inflection Points (Phase Changes):
"""
    
    for _, point in journey['inflection_points'].iterrows():
        narrative += f"\n  {point['month']}: → {point['phase']}"
    
    narrative += "\n\nDetailed Journey:\n"
    narrative += journey['journey'][['month', 'logins_per_day', 'features_used', 'engagement_score', 'phase', 'risk']].to_string(index=False)
    
    return narrative


def save_journey_summary(summary_df, db_path='data/churn_system.db'):
    """Save journey summary to database."""
    
    print(f"\nSaving journey analysis to database...")
    
    conn = sqlite3.connect(db_path)
    
    try:
        summary_df.to_sql('customer_journeys', conn, if_exists='replace', index=False)
        conn.commit()
        print(f"✓ Saved journey analysis for {len(summary_df):,} customers")
    except Exception as e:
        conn.rollback()
        print(f"✗ Error saving journeys: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    # Generate all journeys
    summary = generate_all_journeys()
    
    # Save to database
    if not summary.empty:
        save_journey_summary(summary)
    
    # Example: Print narrative for a specific customer
    print("\n" + "="*70)
    print("EXAMPLE: CUSTOMER JOURNEY NARRATIVE")
    print("="*70)
    narrative = get_journey_narrative(1)
    print(narrative)
    
    print("\n✓ Journey analysis complete!")