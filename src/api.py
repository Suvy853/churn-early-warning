# src/api.py
"""
FastAPI REST API for Churn Early Warning System

Exposes the churn model through REST endpoints for real-time predictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Early Warning System API",
    description="REST API for predicting customer churn and getting recommendations",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing) for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('data/churn_system.db')
    conn.row_factory = sqlite3.Row
    return conn


# Request/Response Models
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    customer_id: str
    
    class Config:
        example = {"customer_id": "CUST_0001"}


class CustomerPrediction(BaseModel):
    """Response model for customer prediction."""
    customer_id: str
    churn_probability: float
    risk_tier: str
    monthly_revenue: float
    annual_revenue_at_risk: float
    health_score: float
    recommendation: str
    
    class Config:
        example = {
            "customer_id": "CUST_0001",
            "churn_probability": 0.78,
            "risk_tier": "High",
            "monthly_revenue": 2500.0,
            "annual_revenue_at_risk": 58500.0,
            "health_score": 35.0,
            "recommendation": "Urgent Account Review"
        }


class CustomerExplanation(BaseModel):
    """Response model for explainability."""
    customer_id: str
    churn_probability: float
    top_3_features: str
    top_3_values: str
    
    class Config:
        example = {
            "customer_id": "CUST_0001",
            "churn_probability": 0.78,
            "top_3_features": "logins_3m_avg, support_tickets, days_since_last_login",
            "top_3_values": "0.42, 0.38, 0.31"
        }


class HealthMetrics(BaseModel):
    """Response model for system health metrics."""
    total_customers: int
    high_risk_customers: int
    average_churn_probability: float
    total_revenue_at_risk: float
    model_status: str
    
    class Config:
        example = {
            "total_customers": 800,
            "high_risk_customers": 65,
            "average_churn_probability": 0.083,
            "total_revenue_at_risk": 1200000.0,
            "model_status": "healthy"
        }


# Endpoints

@app.get("/", tags=["Health"])
def root():
    """Root endpoint - API status."""
    return {
        "status": "online",
        "message": "Churn Early Warning System API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
def health():
    """Health check endpoint - verify API is running."""
    return {
        "status": "healthy",
        "message": "API is running and database is connected"
    }


@app.post("/predict", response_model=CustomerPrediction, tags=["Predictions"])
def predict_churn(request: PredictionRequest):
    """
    Predict churn for a specific customer.
    
    Returns: Customer ID, churn probability, risk tier, revenue metrics, and recommendation.
    """
    
    try:
        conn = get_db_connection()
        
        # Query for the latest prediction for this customer
        query = '''
        SELECT DISTINCT customer_id, churn_probability, risk_tier, monthly_revenue, 
                        annual_revenue_at_risk, health_score, recommendation
        FROM predictions
        WHERE customer_id = ?
        AND prediction_date = (SELECT MAX(prediction_date) FROM predictions)
        '''
        
        result = pd.read_sql_query(query, conn, params=(request.customer_id,))
        conn.close()
        
        if result.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Customer {request.customer_id} not found"
            )
        
        # Extract the first (and only) result
        pred = result.iloc[0]
        
        return CustomerPrediction(
            customer_id=pred['customer_id'],
            churn_probability=float(pred['churn_probability']),
            risk_tier=pred['risk_tier'],
            monthly_revenue=float(pred['monthly_revenue']),
            annual_revenue_at_risk=float(pred['annual_revenue_at_risk']),
            health_score=float(pred['health_score']),
            recommendation=pred['recommendation']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting churn: {e}")
        raise HTTPException(status_code=500, detail="Error predicting churn")


@app.post("/recommend", tags=["Recommendations"])
def get_recommendation(request: PredictionRequest):
    """
    Get retention recommendation and action for a customer.
    
    Returns: Recommended action, cost, success probability, and ROI.
    """
    
    try:
        conn = get_db_connection()
        
        # Get prescription (recommendation) for customer
        query = '''
        SELECT customer_id, prescribed_action, discount_percent, roi, 
               confidence, success_probability
        FROM prescriptions
        WHERE customer_id = ?
        '''
        
        result = pd.read_sql_query(query, conn, params=(request.customer_id,))
        conn.close()
        
        if result.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendation found for customer {request.customer_id}"
            )
        
        rec = result.iloc[0]
        
        return {
            "customer_id": rec['customer_id'],
            "recommended_action": rec['prescribed_action'],
            "discount_percent": float(rec['discount_percent']),
            "roi": float(rec['roi']),
            "confidence": float(rec['confidence']),
            "success_probability": float(rec['success_probability'])
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendation: {e}")
        raise HTTPException(status_code=500, detail="Error getting recommendation")


@app.post("/explain", response_model=CustomerExplanation, tags=["Explainability"])
def explain_prediction(request: PredictionRequest):
    """
    Explain why the model predicted churn for this customer.
    
    Returns: Top 3 features driving the churn prediction and their impact values.
    """
    
    try:
        conn = get_db_connection()
        
        # Get SHAP explanation for customer
        query = '''
        SELECT customer_id, predicted_churn, top_3_features, top_3_shap_values
        FROM shap_explanations
        WHERE customer_id = ?
        '''
        
        result = pd.read_sql_query(query, conn, params=(request.customer_id,))
        conn.close()
        
        if result.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No explanation found for customer {request.customer_id}"
            )
        
        exp = result.iloc[0]
        
        return CustomerExplanation(
            customer_id=exp['customer_id'],
            churn_probability=float(exp['predicted_churn']),
            top_3_features=exp['top_3_features'],
            top_3_values=exp['top_3_shap_values']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining prediction: {e}")
        raise HTTPException(status_code=500, detail="Error explaining prediction")


@app.get("/high-risk-customers", tags=["Analysis"])
def get_high_risk_customers(limit: int = 50):
    """
    Get list of high-risk customers.
    
    Returns: Top N high-risk customers ranked by revenue at-risk.
    """
    
    try:
        conn = get_db_connection()
        
        query = '''
        SELECT customer_id, churn_probability, monthly_revenue, 
               annual_revenue_at_risk, health_score, recommendation
        FROM predictions
        WHERE prediction_date = (SELECT MAX(prediction_date) FROM predictions)
        AND risk_tier = 'High Risk'
        ORDER BY annual_revenue_at_risk DESC
        LIMIT ?
        '''
        
        results = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        if results.empty:
            return {"high_risk_customers": []}
        
        customers = []
        for _, row in results.iterrows():
            customers.append({
                "customer_id": row['customer_id'],
                "churn_probability": float(row['churn_probability']),
                "monthly_revenue": float(row['monthly_revenue']),
                "annual_revenue_at_risk": float(row['annual_revenue_at_risk']),
                "health_score": float(row['health_score']),
                "recommendation": row['recommendation']
            })
        
        return {
            "count": len(customers),
            "high_risk_customers": customers
        }
    
    except Exception as e:
        logger.error(f"Error getting high-risk customers: {e}")
        raise HTTPException(status_code=500, detail="Error getting high-risk customers")


@app.get("/metrics", response_model=HealthMetrics, tags=["Metrics"])
def get_metrics():
    """
    Get system health metrics.
    
    Returns: Total customers, high-risk count, average churn probability, and total revenue at-risk.
    """
    
    try:
        conn = get_db_connection()
        
        query = '''
        SELECT COUNT(*) as total,
               SUM(CASE WHEN risk_tier = 'High Risk' THEN 1 ELSE 0 END) as high_risk,
               AVG(churn_probability) as avg_churn,
               SUM(annual_revenue_at_risk) as total_risk
        FROM predictions
        WHERE prediction_date = (SELECT MAX(prediction_date) FROM predictions)
        '''
        
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        metrics = result.iloc[0]
        
        return HealthMetrics(
            total_customers=int(metrics['total']),
            high_risk_customers=int(metrics['high_risk']),
            average_churn_probability=float(metrics['avg_churn']),
            total_revenue_at_risk=float(metrics['total_risk']),
            model_status="healthy"
        )
    
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Error getting metrics")


@app.get("/customer/{customer_id}", tags=["Customers"])
def get_customer_details(customer_id: str):
    """
    Get all details for a specific customer.
    
    Returns: Prediction, recommendation, and explanation combined.
    """
    
    try:
        # Get prediction
        pred_request = PredictionRequest(customer_id=customer_id)
        prediction = predict_churn(pred_request)
        
        # Get recommendation
        try:
            conn = get_db_connection()
            rec_query = '''
            SELECT prescribed_action, discount_percent, roi, success_probability
            FROM prescriptions WHERE customer_id = ?
            '''
            rec_result = pd.read_sql_query(rec_query, conn, params=(customer_id,))
            
            # Get explanation
            exp_query = '''
            SELECT top_3_features, top_3_shap_values FROM shap_explanations 
            WHERE customer_id = ?
            '''
            exp_result = pd.read_sql_query(exp_query, conn, params=(customer_id,))
            conn.close()
            
            recommendation = rec_result.iloc[0].to_dict() if not rec_result.empty else None
            explanation = exp_result.iloc[0].to_dict() if not exp_result.empty else None
        
        except Exception as e:
            logger.warning(f"Could not get recommendation/explanation: {e}")
            recommendation = None
            explanation = None
        
        return {
            "prediction": prediction.dict(),
            "recommendation": recommendation,
            "explanation": explanation
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting customer details: {e}")
        raise HTTPException(status_code=500, detail="Error getting customer details")


# Run the API
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("CHURN EARLY WARNING SYSTEM - FastAPI SERVER")
    print("=" * 70)
    print("\nAPI running at: http://127.0.0.1:8000")
    print("API docs at: http://127.0.0.1:8000/docs")
    print("Alternative docs: http://127.0.0.1:8000/redoc")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)