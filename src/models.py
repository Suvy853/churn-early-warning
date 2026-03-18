# src/models.py
"""
ML Model Training for Churn Prediction

Builds and evaluates XGBoost model to predict customer churn.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import xgboost as xgb
import joblib
import logging

logging.basicConfig(level=logging.INFO)

def load_and_prepare_data(features_file='data/processed/features.csv'):
    """
    Load features and prepare for modeling.
    
    Args:
        features_file: Path to engineered features CSV
    
    Returns:
        X: Feature matrix (ready for modeling)
        y: Target variable (churn)
        feature_names: List of feature names
    """
    
    print("\n" + "=" * 70)
    print("LOADING & PREPARING DATA")
    print("=" * 70)
    
    # Load features
    print("\nLoading features...")
    df = pd.read_csv(features_file)
    print(f"✓ Loaded {len(df):,} records with {len(df.columns)} columns")
    
    # Separate features (X) and target (y)
    y = df['churned']
    
    # Select features for modeling
    # Exclude: customer_id, year_month, churned, and segment info
    exclude_cols = ['customer_id', 'year_month', 'churned', 
                    'company_size', 'subscription_tier', 'industry', 'monthly_revenue']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()
    
    print(f"\n✓ Selected {len(feature_cols)} features for modeling")
    print(f"  Target distribution:")
    print(f"    Active (0): {(y==0).sum():,} ({(y==0).mean():.1%})")
    print(f"    Churned (1): {(y==1).sum():,} ({(y==1).mean():.1%})")
    
    # Handle any remaining NaNs
    print(f"\nHandling missing values...")
    missing_before = X.isnull().sum().sum()
    X = X.fillna(X.mean())
    missing_after = X.isnull().sum().sum()
    print(f"✓ Filled {missing_before} missing values")
    
    return X, y, feature_cols


def train_test_split_temporal(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets.
    
    Uses random split (for simplicity).
    In production, would use temporal split (older data = train, newer = test).
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Fraction of data for testing
        random_state: For reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    
    print("\n" + "=" * 70)
    print("TRAIN/TEST SPLIT")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain churn rate in both sets
    )
    
    print(f"\nTrain set: {len(X_train):,} records ({len(X_train)/len(X):.1%})")
    print(f"  Churn rate: {y_train.mean():.1%}")
    print(f"\nTest set: {len(X_test):,} records ({len(X_test)/len(X):.1%})")
    print(f"  Churn rate: {y_test.mean():.1%}")
    
    return X_train, X_test, y_train, y_test


def train_xgboost_model(X_train, y_train):
    """
    Train XGBoost model for churn prediction.
    
    XGBoost is chosen because:
    - Excellent performance on structured data
    - Handles non-linear relationships
    - Provides feature importance
    - Fast to train
    """
    
    print("\n" + "=" * 70)
    print("TRAINING XGBOOST MODEL")
    print("=" * 70)
    
    # XGBoost parameters tuned for churn prediction
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'max_depth': 6,  # Depth of trees
        'learning_rate': 0.1,  # How fast to learn
        'n_estimators': 100,  # Number of trees
        'subsample': 0.8,  # Sample 80% of rows for each tree
        'colsample_bytree': 0.8,  # Sample 80% of features for each tree
        'random_state': 42,
    }
    
    print(f"\nTraining with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Train model
    print("\nTraining model...")
    model = xgb.XGBClassifier(**params, verbose=0)
    model.fit(X_train, y_train, 
             eval_set=[(X_train, y_train)],
             verbose=False)
    
    print("✓ Model training complete!")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance on train and test sets.
    
    Key metrics:
    - ROC-AUC: Overall discriminative ability (0.5 = random, 1.0 = perfect)
    - Precision: Of predicted churners, how many actually churned?
    - Recall: Of actual churners, how many did we catch?
    - F1: Harmonic mean of precision and recall
    """
    
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    # Predictions
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    train_precision = precision_score(y_train, y_train_pred)
    test_precision = precision_score(y_test, y_test_pred)
    
    train_recall = recall_score(y_train, y_train_pred)
    test_recall = recall_score(y_test, y_test_pred)
    
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # Print results
    print("\nROC-AUC (primary metric):")
    print(f"  Train: {train_auc:.4f}")
    print(f"  Test:  {test_auc:.4f}")
    print(f"  {'✓ Good generalization' if abs(train_auc - test_auc) < 0.05 else '⚠ Possible overfitting'}")
    
    print("\nPrecision (of predicted churners, how many actually churned?):")
    print(f"  Train: {train_precision:.4f}")
    print(f"  Test:  {test_precision:.4f}")
    
    print("\nRecall (of actual churners, how many did we catch?):")
    print(f"  Train: {train_recall:.4f}")
    print(f"  Test:  {test_recall:.4f}")
    
    print("\nF1-Score (harmonic mean of precision & recall):")
    print(f"  Train: {train_f1:.4f}")
    print(f"  Test:  {test_f1:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"  True Negatives:  {cm[0,0]:,} (correctly predicted active)")
    print(f"  False Positives: {cm[0,1]:,} (incorrectly predicted churn)")
    print(f"  False Negatives: {cm[1,0]:,} (incorrectly predicted active)")
    print(f"  True Positives:  {cm[1,1]:,} (correctly predicted churn)")
    
    return {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'confusion_matrix': cm,
    }


def get_feature_importance(model, feature_names, top_n=15):
    """
    Get feature importance from trained model.
    
    Shows which features the model relies on most to predict churn.
    """
    
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    
    # Get importance scores
    importance_scores = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Most Important Features:")
    print("\n{:<40} {:<10}".format("Feature", "Importance"))
    print("-" * 50)
    for idx, row in importance_df.head(top_n).iterrows():
        print("{:<40} {:<10.4f}".format(row['feature'], row['importance']))
    
    return importance_df


def make_predictions(model, X, feature_names):
    """
    Generate churn predictions for all customers.
    
    Returns both probability (0-1) and binary prediction (0 or 1).
    """
    
    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)
    
    # Get probabilities
    churn_probs = model.predict_proba(X)[:, 1]
    
    # Get binary predictions (threshold = 0.5)
    churn_pred = model.predict(X)
    
    # Create predictions dataframe
    predictions = pd.DataFrame({
        'churn_probability': churn_probs,
        'predicted_churn': churn_pred,
    })
    
    # Create risk tier based on probability
    predictions['risk_tier'] = pd.cut(
        churn_probs,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    print(f"\nPredictions generated for {len(predictions):,} customers")
    print(f"\nRisk Distribution:")
    print(predictions['risk_tier'].value_counts().sort_index())
    
    print(f"\nChurn Probability Statistics:")
    print(f"  Min: {churn_probs.min():.4f}")
    print(f"  Mean: {churn_probs.mean():.4f}")
    print(f"  Median: {np.median(churn_probs):.4f}")
    print(f"  Max: {churn_probs.max():.4f}")
    
    return predictions


def save_model(model, filename='models/churn_model_v1.pkl'):
    """Save trained model to disk."""
    
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    joblib.dump(model, filename)
    print(f"\n✓ Model saved to: {filename}")


if __name__ == "__main__":
    # Full training pipeline
    print("\n" + "=" * 70)
    print("CHURN PREDICTION MODEL TRAINING")
    print("=" * 70)
    
    # Step 1: Load data
    X, y, feature_names = load_and_prepare_data()
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y)
    
    # Step 3: Train model
    model = train_xgboost_model(X_train, y_train)
    
    # Step 4: Evaluate
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Step 5: Feature importance
    importance_df = get_feature_importance(model, feature_names)
    importance_df.to_csv('data/processed/feature_importance.csv', index=False)
    
    # Step 6: Make predictions
    predictions = make_predictions(model, X, feature_names)
    
    # Step 7: Save model
    save_model(model)
    
    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)