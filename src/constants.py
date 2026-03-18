# src/constants.py
# Config and business logic constants for Churn Early Warning System

# ============================================================================
# SUBSCRIPTION TIERS & PRICING
# ============================================================================
TIERS = {
    'Starter': 500,           # $500/month
    'Professional': 2500,     # $2500/month
    'Enterprise': 10000       # $10,000/month
}

# ============================================================================
# CUSTOMER SEGMENTS
# ============================================================================
COMPANY_SIZES = ['small', 'medium', 'large']
INDUSTRIES = ['SaaS', 'Finance', 'Healthcare', 'Retail', 'Other']

# ============================================================================
# DATE RANGE FOR SYNTHETIC DATA
# ============================================================================
DATA_START_DATE = '2023-01'
DATA_END_DATE = '2025-12'
MONTHS_OF_DATA = 36  # 3 years of data

# ============================================================================
# CHURN RATES BY SEGMENT
# ============================================================================
# These are MONTHLY churn rates for synthetic data generation
# Key insight: Smaller companies in starter tier have higher churn
BASE_CHURN_RATES = {
    # Small companies
    ('small', 'Starter'): 0.20,        # 20% monthly churn
    ('small', 'Professional'): 0.12,   # 12% monthly churn
    ('small', 'Enterprise'): 0.05,     # 5% monthly churn
    
    # Medium companies
    ('medium', 'Starter'): 0.15,
    ('medium', 'Professional'): 0.08,
    ('medium', 'Enterprise'): 0.03,
    
    # Large companies
    ('large', 'Starter'): 0.10,
    ('large', 'Professional'): 0.05,
    ('large', 'Enterprise'): 0.01,
}

# ============================================================================
# INDUSTRY CHURN MULTIPLIER
# ============================================================================
# Industries with higher/lower churn than baseline
INDUSTRY_CHURN_MULTIPLIER = {
    'SaaS': 1.0,        # Baseline
    'Finance': 0.7,     # 30% lower churn
    'Healthcare': 0.8,  # 20% lower churn
    'Retail': 1.3,      # 30% higher churn
    'Other': 1.1,       # 10% higher churn
}

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================
AVAILABLE_FEATURES = 10  # Total features available in the platform
FEATURE_ADOPTION_DECAY = 0.05  # Features decline 5% monthly for disengaged customers

# ============================================================================
# ENGAGEMENT THRESHOLDS
# ============================================================================
DAYS_INACTIVE_THRESHOLD = 30   # Mark as "inactive" after 30 days no login
SEVERE_INACTIVE_THRESHOLD = 60  # High-risk threshold (60+ days)

# ============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42  # Change this if you want different data, keep for consistency