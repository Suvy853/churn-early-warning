# B2B Churn Early Warning System

A production-ready machine learning system that predicts customer churn, recommends retention actions, and explains predictions through an interactive dashboard and REST API.

Live Demo: https://churn-early-warning-production.up.railway.app

![alt text](image.png)

---

## Overview

This system solves a common business problem: identifying customers at risk of leaving before they actually churn.

What it does:
1. Predicts which customers are likely to churn
2. Calculates revenue at risk
3. Recommends specific retention actions
4. Tracks customer engagement over time
5. Explains why each prediction was made
6. Sends automated alerts for high-risk customers
7. Monitors model health and detects prediction drift

The system runs daily automatically and serves an interactive 7-tab dashboard plus a REST API for programmatic access.

---

## Key Results

| Metric | Value |
|--------|-------|
| Customers Analyzed | 800 |
| High-Risk Customers Identified | 65 (8.1%) |
| Annual Revenue at Risk | $1.2M |
| Model Performance (ROC-AUC) | 0.82 |
| ROI on Retention Efforts | 5.2x |

---

## Architecture Overview

The system consists of three main components:

### 1. Dashboard (Dash on port 8050)

Interactive web interface with 7 tabs for viewing predictions and insights:

| Tab | Purpose |
|-----|---------|
| Overview | Key metrics and customer risk distribution |
| Priority Customers | Top 50 at-risk customers ranked by revenue impact |
| Risk Analysis | Deep dive into risk factors and correlations |
| Business Impact | ROI scenarios and business case analysis |
| Prescriptions | AI-generated retention recommendations with ROI |
| Journey Analysis | Customer engagement timelines over 36 months |
| Explainability | Top 3 features driving each customer's churn prediction |

### 2. REST API (FastAPI)

Complete REST API with endpoints for programmatic access:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | POST | Get churn prediction for a customer |
| `/recommend` | POST | Get retention recommendation |
| `/explain` | POST | Explain why customer is at risk |
| `/high-risk-customers` | GET | List top high-risk customers |
| `/metrics` | GET | System health metrics |
| `/interventions` | GET | View sent alerts and interventions |
| `/monitoring/health` | GET | Model health score |

### 3. Action Layer

Automated interventions triggered daily:
- Email Alerts sent to sales team for high-risk customers
- Retention Flags created for customer records
- Intervention Tracking logs all actions to database

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.14 |
| ML Model | XGBoost (Binary Classification) |
| Dashboard | Dash with Plotly |
| API Framework | FastAPI with Uvicorn |
| Database | SQLite |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Automation | APScheduler |
| Feature Importance | SHAP |
| Containerization | Docker |
| Cloud Deployment | Railway |

---

## Quick Start

### Option 1: Live Demo (No Installation)

Open in browser: https://churn-early-warning-production.up.railway.app

No setup required.

### Option 2: Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the dashboard:

```bash
python run.py
```

Open in browser: http://127.0.0.1:8050

### Option 3: Run Components Separately

Start Dash dashboard:

```bash
python src/app.py
```

Start FastAPI (in another terminal):

```bash
python src/api.py
```

Dashboard: http://127.0.0.1:8050
API Docs: http://127.0.0.1:8000/docs

### Option 4: Docker

```bash
docker-compose up
```

Open: http://localhost:8050

---

## How It Works

### Daily Pipeline (Runs at 2 AM)

1. Data Ingestion - Load new engagement data
2. Feature Engineering - Transform raw metrics into 21 predictive features
3. Model Scoring - XGBoost predicts churn probability for all customers
4. Revenue Calculation - Estimate revenue at risk
5. Action Generation - Send alerts for high-risk customers
6. Monitoring - Detect prediction drift and anomalies
7. Dashboard Update - Refresh all visualizations

### Model Details

Input Features (21 total):
- Engagement metrics (logins, API calls, features used)
- Trend indicators (3-month moving averages, decay)
- Recency scores (days since last login)
- Risk flags (inactivity, low adoption, support tickets)
- Health scores (composite indicator)

Output:
- Churn Probability: 0-100% likelihood of churning
- Risk Tier: Low (<20%), Medium (20-40%), High (>40%)
- Recommendation: Specific action to take
- Revenue at Risk: 12-month impact if customer churns

---

## API Usage Examples

### Get Customer Prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "CUST_0001"}'
```

Response:

```json
{
  "customer_id": "CUST_0001",
  "churn_probability": 0.081894,
  "risk_tier": "Low Risk",
  "monthly_revenue": 500.0,
  "annual_revenue_at_risk": 1226.82,
  "health_score": 78.0,
  "recommendation": "Monitor"
}
```

### Get Recommendation

```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "CUST_0050"}'
```

### Explain Prediction

```bash
curl -X POST "http://127.0.0.1:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "CUST_0100"}'
```

### Get System Metrics

```bash
curl "http://127.0.0.1:8000/metrics"
```

---

## Monitoring and Alerts

### Model Monitoring

Daily checks for:
- Prediction Drift: Detects if model behavior changed unexpectedly
- Risk Distribution: Alerts if unusual concentration of high-risk customers
- Prediction Statistics: Records average churn, revenue at risk, customer counts

### Action Layer

Automatically triggers:
- Email Alerts to sales team for customers with >40% churn probability
- Retention Flags in database for account review
- Intervention Logs tracking all actions taken

---

## Project Structure

```
churn-early-warning/
├── data/
│   ├── raw/                    Raw CSV files
│   ├── processed/features.csv  Engineered features
│   └── churn_system.db         SQLite database
├── models/
│   └── churn_model_v1.pkl      Trained XGBoost
├── src/
│   ├── app.py                  Dash dashboard
│   ├── api.py                  FastAPI server
│   ├── database.py             Database schema
│   ├── ingest.py               Data loading
│   ├── predict.py              Model scoring
│   ├── scheduler.py            Daily automation
│   ├── actions.py              Alert system
│   ├── monitoring.py           Health checks
│   ├── prescriptions.py        Recommendations
│   ├── journey_analysis.py     Timeline analysis
│   └── shap_analysis.py        Feature importance
├── notebooks/                  Jupyter analysis notebooks
├── Dockerfile                  Container config
├── docker-compose.yml          Docker orchestration
├── run.py                      Dashboard startup
├── requirements.txt            Dependencies
└── README.md                   This file
```

---

## Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.82 |
| Precision | 0.75 |
| Recall | 0.78 |
| F1-Score | 0.76 |

Risk Distribution:
- Low Risk (<20%): 782 customers (97.8%)
- Medium Risk (20-40%): 18 customers (2.3%)
- High Risk (>40%): 5 customers (0.6%)

Note: Only 5 high-risk customers in current data (max churn probability is 53%). Thresholds are calibrated for this distribution.

---

## Deployment

### Local Deployment

```bash
git clone https://github.com/Suvy853/churn-early-warning.git
cd churn-early-warning
pip install -r requirements.txt
python run.py
```

### Docker Deployment

```bash
docker-compose up
```

### Cloud Deployment (Railway)

The system is deployed on Railway.app:

1. Connected to GitHub repository
2. Automatic deployment on every push
3. Dashboard runs 24/7
4. Live at: https://churn-early-warning-production.up.railway.app

---

## Using Your Own Data

To use real customer data instead of synthetic:

Prepare two CSV files:

customers.csv:
```
customer_id, company_name, company_size, industry, subscription_tier, monthly_revenue, onboarding_date, contract_length_months
```

engagement_monthly.csv:
```
customer_id, year_month, active_users, api_calls, logins_per_day, days_since_last_login, features_used_count, support_tickets, engagement_score, churned
```

Update src/ingest.py to load from your source (database, API, etc.)

Then regenerate predictions:

```bash
python src/scheduler.py once
```

---

## FAQ

Q: How often does the model update?
A: Daily at 2:00 AM. Run python src/scheduler.py once to update manually.

Q: Can I use real customer data?
A: Yes, modify src/ingest.py to load from your data source.

Q: How do I add more features?
A: Edit src/features.py to add calculations, then retrain using src/models.py.

Q: Can I deploy to AWS/GCP/Azure?
A: Yes, the Dockerfile works on any cloud platform.

Q: What's the cost to run this?
A: Railway free tier covers this system. $5/month if needed for more resources.

Q: Why isn't the FastAPI running on Railway?
A: The dashboard is the main interface. FastAPI code is available on GitHub for code review. Run locally with: python src/api.py

---

## Project Components

### Data Pipeline
- Generates 800 synthetic customers with 36 months of engagement data
- Engineered 21 predictive features from raw metrics
- Automated daily ingestion via APScheduler

### Machine Learning
- XGBoost binary classifier for churn prediction
- 0.82 ROC-AUC accuracy on test set
- SHAP-based feature importance explanations

### Dashboard
- 7 interactive tabs with Plotly visualizations
- Real-time data queries from SQLite
- Professional business metrics and insights

### API
- RESTful endpoints for all major functions
- FastAPI with automatic Swagger documentation
- Production-ready error handling

### Automation
- Daily 2 AM pipeline execution
- Automated alerts for high-risk customers
- Model monitoring and drift detection

---

## What This Project Demonstrates

End-to-end ML system - Not just a model, a complete production pipeline

Production-ready code - Error handling, logging, documentation

Real business problem - Churn prediction has clear ROI

Multiple skills - Data engineering, ML, backend, DevOps

Cloud deployment - Running live, accessible to stakeholders

API design - RESTful service architecture

Database design - Proper schema with automation

Monitoring - Model health checks in production

---

## Project Status

All 8 phases complete:

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Data Generation | Complete |
| 2 | Feature Engineering | Complete |
| 3 | ML Model Training | Complete |
| 4 | Business Logic | Complete |
| 5 | Database & Automation | Complete |
| 6 | Dashboard | Complete |
| 7 | Advanced Analytics | Complete |
| 8 | Deployment | Complete |

System is production-ready and live on Railway.

---

## Links

| Resource | URL |
|----------|-----|
| GitHub Repo | https://github.com/Suvy853/churn-early-warning |
| Live Demo | https://churn-early-warning-production.up.railway.app |
| Railway Platform | https://railway.app |

---

## Author

Built as a portfolio project demonstrating production ML engineering skills.

---

## License

MIT License - Feel free to use this as a template for your own projects.