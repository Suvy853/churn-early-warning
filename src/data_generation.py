# src/data_generation.py
"""
Synthetic B2B Customer Data Generation

Generates realistic churn scenarios with:
- Engagement decay patterns
- Segment variation
- Seasonality
- Time-in-product effects
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

try:
    from src.constants import (
        TIERS, COMPANY_SIZES, INDUSTRIES, DATA_START_DATE, DATA_END_DATE,
        MONTHS_OF_DATA, BASE_CHURN_RATES, INDUSTRY_CHURN_MULTIPLIER,
        AVAILABLE_FEATURES, FEATURE_ADOPTION_DECAY, RANDOM_SEED
    )
except ImportError:
    from constants import (
        TIERS, COMPANY_SIZES, INDUSTRIES, DATA_START_DATE, DATA_END_DATE,
        MONTHS_OF_DATA, BASE_CHURN_RATES, INDUSTRY_CHURN_MULTIPLIER,
        AVAILABLE_FEATURES, FEATURE_ADOPTION_DECAY, RANDOM_SEED
    )

# Set seeds for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def generate_customer_base(n_customers=800):
    """Generate static customer attributes."""
    print(f"Generating {n_customers} unique customers...")
    
    company_names = [f"Company_{i}" for i in range(1, n_customers + 1)]
    
    data = {
        'customer_id': [f'CUST_{i:04d}' for i in range(1, n_customers + 1)],
        'company_name': company_names,
        'company_size': np.random.choice(COMPANY_SIZES, n_customers, p=[0.50, 0.35, 0.15]),
        'industry': np.random.choice(INDUSTRIES, n_customers, p=[0.30, 0.20, 0.15, 0.20, 0.15]),
        'subscription_tier': np.random.choice(
            list(TIERS.keys()), 
            n_customers,
            p=[0.50, 0.35, 0.15]
        ),
    }
    
    customers = pd.DataFrame(data)
    customers['monthly_revenue'] = customers['subscription_tier'].map(TIERS)
    
    customers['onboarding_date'] = [
        datetime(2023, 1, 1) + timedelta(days=int(np.random.uniform(0, 730)))
        for _ in range(n_customers)
    ]
    
    contract_lengths = []
    for tier in customers['subscription_tier']:
        if tier == 'Starter':
            contract_lengths.append(np.random.randint(6, 13))
        elif tier == 'Professional':
            contract_lengths.append(np.random.randint(12, 25))
        else:
            contract_lengths.append(np.random.randint(24, 37))
    
    customers['contract_length_months'] = contract_lengths
    
    print(f"✓ Generated {len(customers)} customers")
    return customers


def calculate_churn_probability(row, month_index):
    """Calculate probability of churn for a customer in a given month."""
    
    segment_key = (row['company_size'], row['subscription_tier'])
    base_rate = BASE_CHURN_RATES[segment_key]
    industry_mult = INDUSTRY_CHURN_MULTIPLIER[row['industry']]
    
    month_of_year = (month_index % 12) + 1
    if month_of_year in [1, 4, 10, 12]:
        seasonality = 1.15
    else:
        seasonality = 0.85
    
    months_since_onboard = month_index - (
        (row['onboarding_date'].year - 2023) * 12 +
        (row['onboarding_date'].month - 1)
    )
    
    if months_since_onboard < 0:
        return 0.0
    elif months_since_onboard < 2:
        time_effect = 1.5
    elif months_since_onboard < 6:
        time_effect = 1.2
    else:
        time_effect = 1.0
    
    final_prob = base_rate * industry_mult * seasonality * time_effect
    return min(final_prob, 0.50)


def generate_monthly_engagement(customers, month_index, month_str):
    """Generate monthly usage metrics for all customers."""
    n = len(customers)
    engagement = pd.DataFrame()
    
    engagement['customer_id'] = customers['customer_id']
    engagement['year_month'] = month_str
    
    base_active_users = customers['subscription_tier'].map({
        'Starter': 3,
        'Professional': 10,
        'Enterprise': 50,
    })
    
    base_api_calls = customers['subscription_tier'].map({
        'Starter': 1000,
        'Professional': 10000,
        'Enterprise': 100000,
    })
    
    engagement['active_users'] = (
        base_active_users * np.random.normal(1.0, 0.3, n)
    ).astype(int).clip(lower=1)
    
    engagement['api_calls'] = (
        base_api_calls * np.random.normal(1.0, 0.4, n)
    ).astype(int).clip(lower=100)
    
    engagement['logins_per_day'] = np.random.uniform(0.2, 5.0, n)
    engagement['days_since_last_login'] = np.random.randint(0, 60, n)
    engagement['features_used_count'] = np.random.randint(1, AVAILABLE_FEATURES + 1, n)
    engagement['support_tickets'] = np.random.randint(0, 5, n)
    
    engagement['engagement_score'] = (
        (engagement['active_users'] / engagement['active_users'].max() * 25) +
        (engagement['logins_per_day'] / 5.0 * 25) +
        (engagement['features_used_count'] / AVAILABLE_FEATURES * 25) +
        ((60 - engagement['days_since_last_login']) / 60.0 * 25)
    ).round(1).clip(lower=0, upper=100)
    
    return engagement


def assign_churn_labels(engagement_data, customers, month_index, month_str):
    """Assign churn labels based on probabilities."""
    engagement_data['churned'] = 0
    
    for idx, row in customers.iterrows():
        churn_prob = calculate_churn_probability(row, month_index)
        if np.random.random() < churn_prob:
            engagement_data.loc[
                engagement_data['customer_id'] == row['customer_id'],
                'churned'
            ] = 1
    
    return engagement_data


def generate_dataset(n_customers=800):
    """Main function: Generate complete 3-year synthetic dataset."""
    
    print("=" * 70)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 70)
    
    customers = generate_customer_base(n_customers=n_customers)
    engagement_data = []
    
    print(f"\nGenerating {MONTHS_OF_DATA} months of engagement data...")
    for month_index in range(MONTHS_OF_DATA):
        start_date = datetime.strptime(DATA_START_DATE, '%Y-%m')
        current_date = start_date + timedelta(days=30 * month_index)
        month_str = current_date.strftime('%Y-%m')
        
        if month_index % 6 == 0:
            print(f"  {month_str}...", end=' ', flush=True)
        
        monthly_eng = generate_monthly_engagement(customers, month_index, month_str)
        monthly_eng = assign_churn_labels(monthly_eng, customers, month_index, month_str)
        engagement_data.append(monthly_eng)
        
        if month_index % 6 == 5:
            print("✓")
    
    engagement_df = pd.concat(engagement_data, ignore_index=True)
    
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Customers:              {len(customers):,}")
    print(f"Total records:          {len(engagement_df):,}")
    print(f"Time period:            {MONTHS_OF_DATA} months")
    print(f"Total churn events:     {engagement_df['churned'].sum():,}")
    print(f"Overall churn rate:     {engagement_df['churned'].mean():.2%}")
    
    return customers, engagement_df
    print("=" * 70)
    
    return customers, engagement_df


if __name__ == "__main__":
    customers, engagement = generate_dataset(n_customers=800)
    customers.to_csv('data/raw/customers.csv', index=False)
    engagement.to_csv('data/raw/engagement_monthly.csv', index=False)
    print("\n✓ Data saved successfully!")
    print(f"  - data/raw/customers.csv ({len(customers)} rows)")
    print(f"  - data/raw/engagement_monthly.csv ({len(engagement)} rows)")