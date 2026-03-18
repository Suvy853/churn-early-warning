# src/shap_analysis.py
"""
SHAP Explainability Analysis

Generates SHAP (SHapley Additive exPlanations) values to explain:
- Why the model predicted a specific churn probability for a customer
- Which features contributed most to the prediction
- Feature impact direction (positive/negative)
"""

import pandas as pd
import sqlite3
import pickle
import numpy as np
import shap
import logging

logging.basicConfig(level=logging.INFO)


def load_model(model_path='models/churn_model_v1.pkl'):
    """Load the trained XGBoost model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def get_db_connection(db_path='data/churn_system.db'):
    """Get database connection."""
    conn = sqlite3.connect(db_path)
    return conn


def get_latest_features(db_path='data/churn_system.db'):
    """Get latest features from CSV file - one row per customer."""
    import os
    
    # Load features from CSV
    features_path = 'data/processed/features.csv'
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    df = pd.read_csv(features_path)
    
    # Get the maximum year_month
    max_month = df['year_month'].max()
    
    # Filter to ONLY the latest month
    df_latest = df[df['year_month'] == max_month].copy()
    
    print(f"Loaded {len(df_latest)} customers from latest month: {max_month}")
    
    # Get latest predictions to match with features
    conn = get_db_connection(db_path)
    
    query = '''
    SELECT DISTINCT customer_id, churn_probability, health_score, recommendation, risk_tier
    FROM predictions
    WHERE prediction_date = (SELECT MAX(prediction_date) FROM predictions)
    '''
    
    predictions = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Merge features with latest predictions on customer_id
    merged = df_latest.merge(predictions, on='customer_id', how='inner')

    # Ensure feature names match model training (avoid _x/_y suffixes)
    if 'health_score_x' in merged.columns and 'health_score_y' in merged.columns:
        merged = merged.rename(columns={'health_score_x': 'health_score'})
        merged = merged.drop(columns=['health_score_y'])
    
    print(f"Merged data has {len(merged)} rows")
    
    return merged


def get_customer_features(customer_id, db_path='data/churn_system.db'):
    """Get features for a specific customer."""
    import os
    
    # Load features from CSV
    features_path = 'data/processed/features.csv'
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    df = pd.read_csv(features_path)
    
    # Get latest month for this customer
    df_latest = df[(df['customer_id'] == customer_id) & (df['year_month'] == df['year_month'].max())]
    
    if df_latest.empty:
        return None
    
    return df_latest.iloc[0]


def prepare_features_for_shap(df):
    """Prepare features in the format the model expects."""
    feature_cols = [
        'logins_per_day', 'api_calls', 'features_used_count', 'days_inactive',
        'support_tickets', 'engagement_score', 'logins_3m_avg', 'api_calls_3m_avg',
        'features_3m_avg', 'logins_trend_3m', 'engagement_decay', 'recency_score',
        'feature_adoption_rate', 'is_high_adoption', 'is_low_adoption',
        'flag_no_login_30days', 'flag_no_login_60days', 'flag_low_feature_adoption',
        'flag_high_support_tickets', 'flag_declining_engagement', 'risk_flag_count',
        'health_score'
    ]
    
    # Filter to only existing columns
    existing_cols = [col for col in feature_cols if col in df.columns]
    
    return df[existing_cols]


def generate_shap_values(model, X):
    """Generate SHAP values for predictions."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # For binary classification, get positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    return shap_values, explainer


def get_customer_explanation(customer_id, model_path='models/churn_model_v1.pkl', db_path='data/churn_system.db'):
    """
    Generate SHAP explanation for a specific customer.
    
    Returns:
        Dictionary with explanation data
    """
    
    print(f"\nGenerating SHAP explanation for customer {customer_id}...")
    
    # Load model
    model = load_model(model_path)
    
    # Get customer features
    customer_row = get_customer_features(customer_id, db_path)
    if customer_row is None:
        return None
    
    # Get all data for SHAP (background)
    all_data = get_latest_features(db_path)
    X_all = prepare_features_for_shap(all_data)
    
    # Prepare customer data
    X_customer = prepare_features_for_shap(pd.DataFrame([customer_row]))
    
    # Generate SHAP values
    shap_values, explainer = generate_shap_values(model, X_all)
    
    # Get customer's index in the data
    customer_mask = (all_data['customer_id'] == customer_id)
    customer_idx = all_data[customer_mask].index[0] if customer_mask.any() else None
    
    if customer_idx is None:
        return None
    
    # Get customer's SHAP values
    customer_shap = shap_values[customer_idx]
    
    # Get feature names
    feature_names = X_all.columns.tolist()
    
    # Get base value (expected model output)
    base_value = explainer.expected_value
    
    # Get customer's predicted churn probability
    predicted_churn = all_data[customer_mask].iloc[0]['churn_probability']
    
    # Create dataframe with feature contributions
    contributions = pd.DataFrame({
        'feature': feature_names,
        'value': X_customer.iloc[0].values,
        'shap_value': customer_shap,
        'abs_shap_value': np.abs(customer_shap)
    })
    
    # Sort by absolute SHAP value (most important first)
    contributions = contributions.sort_values('abs_shap_value', ascending=False)
    
    return {
        'customer_id': customer_id,
        'predicted_churn': predicted_churn,
        'base_value': base_value,
        'contributions': contributions,
        'shap_values': shap_values,
        'explainer': explainer,
        'X_all': X_all
    }


def get_top_features_for_customer(customer_id, top_n=10, model_path='models/churn_model_v1.pkl', db_path='data/churn_system.db'):
    """Get top N features contributing to customer's churn prediction."""
    
    explanation = get_customer_explanation(customer_id, model_path, db_path)
    if explanation is None:
        return None
    
    top_features = explanation['contributions'].head(top_n)
    
    return {
        'customer_id': customer_id,
        'predicted_churn': explanation['predicted_churn'],
        'top_features': top_features
    }


def generate_all_customer_explanations(model_path='models/churn_model_v1.pkl', db_path='data/churn_system.db'):
    """Generate SHAP explanations for all customers."""
    
    print("\n" + "=" * 70)
    print("GENERATING SHAP EXPLANATIONS")
    print("=" * 70)
    
    # Load model
    model = load_model(model_path)
    
    # Get all data
    all_data = get_latest_features(db_path)
    X_all = prepare_features_for_shap(all_data)
    
    print(f"\nGenerating SHAP values for {len(X_all)} customers...")
    
    # Generate SHAP values once for all customers
    shap_values, explainer = generate_shap_values(model, X_all)
    
    base_value = explainer.expected_value
    
    # Create summary: top 3 features for each customer
    feature_names = X_all.columns.tolist()
    
    summaries = []
    
    # Reset index to match with shap_values
    all_data_reset = all_data.reset_index(drop=True)
    X_all_reset = X_all.reset_index(drop=True)
    
    for idx in range(len(X_all_reset)):
        customer_id = all_data_reset.iloc[idx]['customer_id']
        customer_shap = shap_values[idx]
        predicted_churn = all_data_reset.iloc[idx]['churn_probability']
        
        # Get top 3 contributing features
        contributions = pd.DataFrame({
            'feature': feature_names,
            'value': X_all_reset.iloc[idx].values,
            'shap_value': customer_shap,
            'abs_shap_value': np.abs(customer_shap)
        })
        
        contributions = contributions.sort_values('abs_shap_value', ascending=False)
        top_3 = contributions.head(3)
        
        # Create text summary
        top_features_text = ', '.join([f"{row['feature']}" for _, row in top_3.iterrows()])
        top_values = ', '.join([f"{row['shap_value']:.3f}" for _, row in top_3.iterrows()])
        
        summaries.append({
            'customer_id': customer_id,
            'predicted_churn': predicted_churn,
            'top_3_features': top_features_text,
            'top_3_shap_values': top_values
        })
    
    summary_df = pd.DataFrame(summaries)
    
    print(f"\n✓ Generated SHAP explanations for {len(summary_df)} customers")
    
    return summary_df


def save_shap_summary(summary_df, db_path='data/churn_system.db'):
    """Save SHAP summary to database."""
    
    print(f"\nSaving SHAP summary to database...")
    
    conn = get_db_connection(db_path)
    
    try:
        summary_df.to_sql('shap_explanations', conn, if_exists='replace', index=False)
        conn.commit()
        print(f"✓ Saved SHAP explanations for {len(summary_df)} customers")
    except Exception as e:
        conn.rollback()
        print(f"✗ Error saving SHAP explanations: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    # Generate SHAP explanations for all customers
    summary = generate_all_customer_explanations()
    
    # Save to database
    if summary is not None:
        save_shap_summary(summary)
        
        print(f"\n" + "=" * 70)
        print("EXAMPLE: TOP FEATURES FOR FIRST 5 CUSTOMERS")
        print("=" * 70)
        print(summary.head(5).to_string(index=False))
        
        print(f"\n✓ SHAP analysis complete!")