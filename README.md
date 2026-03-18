# B2B Churn Early Warning System

A production-ready machine learning system that predicts customer churn, recommends retention actions, and explains predictions through an interactive dashboard.

Live Demo: https://churn-early-warning-production.up.railway.app

<img width="1899" height="868" alt="image" src="https://github.com/user-attachments/assets/1d53cbb1-c97a-476e-bf55-d4f2b758b12c" />

## Overview

This system solves a common business problem: identifying customers at risk of leaving before they actually churn.

What it does:
1. Predicts which customers are likely to churn
2. Calculates revenue at risk
3. Recommends specific retention actions
4. Tracks customer engagement over time
5. Explains why each prediction was made

The system runs daily automatically and serves an interactive 7-tab dashboard for viewing predictions and insights.

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

## Dashboard Features

The dashboard contains 7 interactive tabs designed for different stakeholders.

**Tab 1: Overview** displays key business metrics including total customers, high-risk count, and revenue at risk. It shows risk distribution pie charts and churn probability histograms.

**Tab 2: Priority Customers** lists the top 50 customers at highest risk ranked by revenue impact. Each customer shows churn probability, health score, and recommended actions.

**Tab 3: Risk Analysis** provides deep-dive analysis into risk factors. It shows revenue at risk broken down by tier and analyzes the relationship between health score and churn probability.

**Tab 4: Business Impact** answers business questions like "If we retain 20% of high-risk customers, how much revenue do we save?" It shows ROI scenarios for different retention rates.

**Tab 5: Prescriptions** recommends specific retention actions for each high-risk customer. Each recommendation includes the action type, estimated cost, success probability, and expected ROI.

**Tab 6: Journey Analysis** tracks customer engagement timelines across 36 months. It shows when customers became at-risk and identifies critical inflection points in their engagement.

**Tab 7: Explainability** reveals the top 3 features driving each customer's churn prediction. It explains which factors matter most for understanding why the model made its prediction.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.14 |
| ML Model | XGBoost (Binary Classification) |
| Dashboard | Dash with Plotly |
| Database | SQLite |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Automation | APScheduler |
| Feature Importance | SHAP |
| Containerization | Docker |
| Cloud Deployment | Railway |

---

## Quick Start

### Option 1: Live Demo (No Installation)

Open this link in your browser: https://churn-early-warning-production.up.railway.app

The demo runs on Railway cloud servers and is accessible 24/7. No setup required.

### Option 2: Run Locally

First, install all dependencies:

```bash
pip install -r requirements.txt
```

Then start the dashboard:

```bash
python src/app.py
```

Open your browser to: http://127.0.0.1:8050

### Option 3: Docker

Run with Docker Compose:

```bash
docker-compose up
```

Open your browser to: http://localhost:8050

---

## Project Structure

The project is organized into logical directories for data, models, code, and documentation.

**data/** contains all customer and engagement data. The raw subfolder has original CSV files. The processed subfolder contains engineered features. The churn_system.db file is the SQLite database.

**models/** stores the trained machine learning model. The churn_model_v1.pkl file contains the XGBoost model ready for predictions.

**src/** contains all application code. The app.py file is the main Dash dashboard. Supporting files handle database operations, data ingestion, model scoring, scheduling, recommendations, journey analysis, and feature importance.

**notebooks/** contains Jupyter notebooks for data science analysis and model development.

Additional files at the root include Dockerfile for containerization, docker-compose.yml for orchestration, and requirements.txt for Python dependencies.

---

## How It Works

### Step 1: Data Collection

The system collects engagement data from 800 B2B customers across 36 months of history. Data includes metrics like monthly logins, API calls, features used, support tickets, and overall engagement scores.

### Step 2: Feature Engineering

Raw engagement metrics are transformed into 21 engineered features. These include 3-month moving averages, engagement decay calculations, feature adoption rates, and various risk flags for critical events.

### Step 3: Model Training

An XGBoost model is trained to predict churn probability (0-100%) for each customer. The model achieves 0.82 ROC-AUC accuracy on test data, balancing precision and recall.

### Step 4: Business Logic

Revenue at-risk is calculated as: Monthly Revenue × Churn Probability × 3 months. Customers are classified into three risk tiers: Low (under 30%), Medium (30-60%), and High (over 60%).

### Step 5: Automated Pipeline

APScheduler runs the entire pipeline daily at 2:00 AM. It ingests new data, engineers features, scores all customers, calculates revenue at-risk, generates recommendations, analyzes journeys, and updates the dashboard.

### Step 6: Interactive Dashboard

Dash serves the 7-tab dashboard with real-time data updated every 5 minutes. All visualizations query the SQLite database directly.

---

## Daily Automation

The system automatically executes the complete pipeline every day at 2:00 AM.

The pipeline performs these tasks in sequence: load new engagement data, engineer 21 features, score all 800 customers with the XGBoost model, calculate revenue at-risk, generate retention recommendations, analyze customer journeys, create feature importance explanations, and update the live dashboard.

To run the automation manually at any time, use this command:

```bash
python src/scheduler.py once
```

---

## Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.82 |
| Precision (High Risk) | 0.75 |
| Recall (High Risk) | 0.78 |
| F1-Score | 0.76 |

The model distributes customers across risk tiers effectively. Low risk customers (under 30% churn probability) comprise 642 customers or 80.3%. Medium risk customers (30-60% churn probability) comprise 93 customers or 11.6%. High risk customers (over 60% churn probability) comprise 65 customers or 8.1%.

---

## Deployment

The system is currently deployed on Railway.app, a cloud platform that automatically deploys from GitHub repositories. This ensures the latest code is always running on production servers.

The application runs 24/7 on Railway's cloud infrastructure without requiring any manual intervention. The live dashboard is accessible at https://churn-early-warning-production.up.railway.app from any browser worldwide.

Other deployment options are available including AWS, Google Cloud, Azure, and Heroku. The Dockerfile included in the repository facilitates easy deployment to any cloud platform.

---

## Main Application Files

| File | Purpose |
|------|---------|
| app.py | Main Dash dashboard with 7 interactive tabs |
| database.py | SQLite schema setup and initialization |
| ingest.py | Data loading and validation pipeline |
| predict.py | XGBoost model scoring |
| scheduler.py | Daily automation orchestration |
| prescriptions.py | Retention recommendation engine |
| journey_analysis.py | Customer timeline and phase analysis |
| shap_analysis.py | Feature importance calculation |

---

## Data Details

### Customer Data

The system analyzes 800 synthetic B2B companies distributed across three subscription tiers. Starter tier costs $500 per month. Professional tier costs $2,500 per month. Enterprise tier costs $10,000 per month.

The dataset spans 36 months of engagement history from January 2023 to December 2025.

### Engagement Metrics

Raw engagement data includes monthly logins, API calls per month, count of features used, days since last login, support tickets opened, and an overall engagement score.

### Engineered Features

21 features are derived from raw metrics including 3-month moving averages, trend calculations, engagement decay metrics, feature adoption rates, and risk flags for critical events like prolonged inactivity or high support ticket volume.

---

## Using Your Own Data

The system currently uses synthetic CSV data for demonstration purposes. You can easily replace this with your own real customer data.

To integrate real data, modify the data source in src/ingest.py. Replace the CSV loading code with a connection to your actual data source such as PostgreSQL, Snowflake, or BigQuery.

Your data must include these required columns: customer_id, year_month (YYYY-MM format), logins_per_day, api_calls, features_used_count, days_since_last_login, support_tickets, and engagement_score.

The rest of the pipeline remains unchanged, allowing you to swap data sources without modifying any other components.

---

## Requirements

This project requires Python 3.10 or higher to run successfully.

Install all dependencies using pip:

```bash
pip install -r requirements.txt
```

The requirements.txt file contains all necessary packages including Dash, XGBoost, Pandas, NumPy, Plotly, APScheduler, and others.

---

## Why This Project

This project demonstrates a complete end-to-end machine learning system from data collection to cloud deployment. It showcases production-ready code with proper error handling, logging, and clean architecture patterns.

The system solves a real business problem with quantifiable ROI. It combines multiple technical skills including feature engineering, model development, data automation, and cloud deployment. It includes a live demo link that recruiters can access immediately to see the working system.

The project is also easily adaptable to real data sources, showing how to build scalable ML pipelines that work in production environments.

---

## FAQ

**Q: How often is the data updated?**

A: Data updates daily at 2:00 AM automatically. You can also run the pipeline manually at any time with python src/scheduler.py once.

**Q: Can I use real customer data instead of synthetic?**

A: Yes. Replace the data source in src/ingest.py with your actual customer data. See the "Using Your Own Data" section.

**Q: How do I change the automation schedule time?**

A: Edit src/scheduler.py and modify the start_scheduler() function call to specify a different time.

**Q: Can I add more features to improve the model?**

A: Yes. Add new calculations to src/features.py and retrain the model using the Jupyter notebooks provided.

**Q: Can I deploy to other cloud platforms?**

A: Yes. The Dockerfile is included in the repository and can be deployed to AWS, Google Cloud, Azure, or any other Docker-compatible platform.

---

## Important Links

| Resource | URL |
|----------|-----|
| GitHub Repository | https://github.com/Suvy853/churn-early-warning |
| Live Dashboard | https://churn-early-warning-production.up.railway.app |
| Railway Platform | https://railway.app |
| Dash Documentation | https://dash.plotly.com |
| XGBoost Documentation | https://xgboost.readthedocs.io |

---

## Project Completion Status

| Phase | Task | Status |
|-------|------|--------|
| 1 | Data Generation | Complete |
| 2 | Feature Engineering | Complete |
| 3 | ML Model Training | Complete |
| 4 | Business Logic | Complete |
| 5 | Database and Automation | Complete |
| 6 | Interactive Dashboard | Complete |
| 7 | Advanced Analytics | Complete |
| 8 | Deployment and Documentation | Complete |

All phases are complete and the system is production-ready with live deployment accessible at https://churn-early-warning-production.up.railway.app
