# src/app.py
"""
Dash Dashboard for Churn Early Warning System

Multi-page interactive dashboard for viewing:
- Customer risk predictions
- Priority list for retention
- Business metrics
- Revenue impact analysis
"""

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Database connection
def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('data/churn_system.db')
    conn.row_factory = sqlite3.Row
    return conn


def get_latest_predictions():
    """Get latest predictions from database."""
    conn = get_db_connection()
    query = '''
    SELECT * FROM predictions
    WHERE prediction_date = (SELECT MAX(prediction_date) FROM predictions)
    ORDER BY risk_rank ASC
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return pd.DataFrame()
    
    return df


def get_business_metrics():
    """Calculate key business metrics."""
    df = get_latest_predictions()
    
    if df.empty:
        return {
            'total_customers': 0,
            'high_risk': 0,
            'total_revenue_at_risk': 0,
            'avg_churn_prob': 0,
            'revenue_at_risk_high_risk': 0
        }
    
    return {
        'total_customers': len(df),
        'high_risk': len(df[df['risk_tier'] == 'High Risk']),
        'total_revenue_at_risk': df['annual_revenue_at_risk'].sum(),
        'avg_churn_prob': df['churn_probability'].mean(),
        'revenue_at_risk_high_risk': df[df['risk_tier'] == 'High Risk']['annual_revenue_at_risk'].sum()
    }


# ============================================================================
# HELPER FUNCTIONS - Prescriptions & Journey Analysis
# ============================================================================

def get_prescriptions():
    """Get prescriptions from database."""
    conn = get_db_connection()
    query = 'SELECT * FROM prescriptions ORDER BY roi DESC'
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return pd.DataFrame()
    
    return df


def get_journeys():
    """Get customer journeys from database."""
    conn = get_db_connection()
    query = 'SELECT * FROM customer_journeys'
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return pd.DataFrame()
    
    return df


def get_journey_details(customer_id):
    """Get detailed journey for a specific customer."""
    conn = get_db_connection()
    query = '''
    SELECT * FROM engagement_raw
    WHERE customer_id = ?
    ORDER BY year_month ASC
    '''
    df = pd.read_sql_query(query, conn, params=(customer_id,))
    conn.close()
    
    return df


def get_shap_explanations():
    """Get SHAP explanations from database."""
    conn = get_db_connection()
    query = 'SELECT * FROM shap_explanations'
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return pd.DataFrame()
    
    return df


# Define app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Churn Early Warning System", className="mb-4 mt-4", style={'color': '#1f77b4'})
        ])
    ]),
    
    dcc.Tabs(id='tabs', value='tab-1', children=[
        # ====================================================================
        # TAB 1 - OVERVIEW
        # ====================================================================
        dcc.Tab(label='Overview', value='tab-1', children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Total Customers", className="text-muted"),
                                html.H2(id='metric-customers', children="0", style={'color': '#1f77b4'})
                            ])
                        ], className="mb-3")
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("High Risk", className="text-muted"),
                                html.H2(id='metric-high-risk', children="0", style={'color': 'red'})
                            ])
                        ], className="mb-3")
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Annual Revenue at Risk", className="text-muted"),
                                html.H2(id='metric-revenue', children="$0M", style={'color': 'orange'})
                            ])
                        ], className="mb-3")
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Avg Churn Prob", className="text-muted"),
                                html.H2(id='metric-churn-prob', children="0%", style={'color': '#ff7f0e'})
                            ])
                        ], className="mb-3")
                    ], width=3)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='risk-distribution-chart')
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id='recommendation-chart')
                    ], width=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='churn-probability-dist')
                    ])
                ])
            ], fluid=True, className="mt-4")
        ]),
        
        # ====================================================================
        # TAB 2 - PRIORITY CUSTOMERS
        # ====================================================================
        dcc.Tab(label='Priority Customers', value='tab-2', children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3("Top 50 Customers at Risk", className="mt-3 mb-3")
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.Div(id='priority-table')
                    ])
                ], className="mt-3")
            ], fluid=True)
        ]),
        
        # ====================================================================
        # TAB 3 - RISK ANALYSIS
        # ====================================================================
        dcc.Tab(label='Risk Analysis', value='tab-3', children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3("Customer Risk Distribution", className="mt-3 mb-3")
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='revenue-at-risk-tier')
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id='churn-prob-scatter')
                    ], width=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='health-score-dist')
                    ])
                ])
            ], fluid=True)
        ]),
        
        # ====================================================================
        # TAB 4 - BUSINESS IMPACT
        # ====================================================================
        dcc.Tab(label='Business Impact', value='tab-4', children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3("Retention Strategy & ROI", className="mt-3 mb-3")
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='retention-scenarios')
                    ], width=6),
                    dbc.Col([
                        html.Div(id='business-metrics-text')
                    ], width=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='recommendation-breakdown')
                    ])
                ])
            ], fluid=True)
        ]),
        
        # ====================================================================
        # TAB 5 - PRESCRIPTIONS
        # ====================================================================
        dcc.Tab(label='Prescriptions', value='tab-5', children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3("Recommended Actions with ROI", className="mt-3 mb-3")
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Portfolio Impact"),
                                html.Div(id='prescription-metrics')
                            ])
                        ])
                    ], width=12)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='roi-chart')
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id='action-type-chart')
                    ], width=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H5("Top Actions by ROI", className="mt-4 mb-3")
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.Div(id='prescriptions-table')
                    ])
                ])
            ], fluid=True)
        ]),
        
        # ====================================================================
        # TAB 6 - JOURNEY ANALYSIS
        # ====================================================================
        dcc.Tab(label='Journey Analysis', value='tab-6', children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3("Customer Journey Timelines", className="mt-3 mb-3")
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Customer:"),
                        dcc.Dropdown(
                            id='customer-dropdown',
                            options=[],
                            value=None,
                            placeholder="Choose a customer",
                            style={'width': '100%'}
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Journey Summary"),
                                html.Div(id='journey-summary')
                            ])
                        ])
                    ], width=8)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='journey-chart')
                    ])
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H5("Journey Details", className="mt-4 mb-3")
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.Div(id='journey-table')
                    ])
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H5("Phase Distribution", className="mt-4 mb-3")
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='phase-distribution-chart')
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id='risk-timeline-chart')
                    ], width=6)
                ])
            ], fluid=True)
        ]),
        
        # ====================================================================
        # TAB 7 - SHAP EXPLAINABILITY
        # ====================================================================
        dcc.Tab(label='Explainability (SHAP)', value='tab-7', children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3("Feature Importance & Model Explanations", className="mt-3 mb-3")
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Customer:"),
                        dcc.Dropdown(
                            id='shap-customer-dropdown',
                            options=[],
                            value=None,
                            placeholder="Choose a customer",
                            style={'width': '100%'}
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Prediction Summary"),
                                html.Div(id='shap-summary')
                            ])
                        ])
                    ], width=8)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        html.H5("Top Features Contributing to Prediction", className="mb-3")
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.Div(id='shap-explanation-table')
                    ])
                ])
            ], fluid=True)
        ])
    ]),
    
    # Auto-refresh every 5 minutes
    dcc.Interval(id='interval-component', interval=300*1000, n_intervals=0)
], fluid=True, className="mt-2")


# ============================================================================
# CALLBACKS - METRICS & CHARTS
# ============================================================================

@app.callback(
    [Output('metric-customers', 'children'),
     Output('metric-high-risk', 'children'),
     Output('metric-revenue', 'children'),
     Output('metric-churn-prob', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_metrics(n):
    metrics = get_business_metrics()
    
    return (
        f"{metrics['total_customers']:,}",
        f"{metrics['high_risk']:,}",
        f"${metrics['total_revenue_at_risk']/1_000_000:.1f}M",
        f"{metrics['avg_churn_prob']*100:.1f}%"
    )


@app.callback(
    Output('risk-distribution-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_risk_dist(n):
    df = get_latest_predictions()
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available")
    
    risk_counts = df['risk_tier'].value_counts()
    colors = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
    
    fig = go.Figure(data=[
        go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.3,
            marker=dict(colors=[colors.get(tier, 'gray') for tier in risk_counts.index])
        )
    ])
    
    fig.update_layout(title='Customers by Risk Tier', height=400, showlegend=True)
    return fig


@app.callback(
    Output('recommendation-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_rec_chart(n):
    df = get_latest_predictions()
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available")
    
    rec_counts = df['recommendation'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(x=rec_counts.index, y=rec_counts.values, marker=dict(color='steelblue'))
    ])
    
    fig.update_layout(
        title='Customers by Recommendation',
        xaxis_title='Recommendation',
        yaxis_title='Count',
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig


@app.callback(
    Output('churn-probability-dist', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_churn_dist(n):
    df = get_latest_predictions()
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available")
    
    fig = go.Figure(data=[
        go.Histogram(x=df['churn_probability'], nbinsx=30, marker=dict(color='steelblue'))
    ])
    
    fig.update_layout(
        title='Distribution of Churn Probabilities',
        xaxis_title='Churn Probability',
        yaxis_title='Number of Customers',
        height=400
    )
    
    return fig


@app.callback(
    Output('priority-table', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_priority_table(n):
    df = get_latest_predictions()
    
    if df.empty:
        return dbc.Alert("No predictions available yet. Run the pipeline first.", color="warning")
    
    top_50 = df.nlargest(50, 'revenue_at_risk')[
        ['customer_id', 'monthly_revenue', 'churn_probability', 
         'health_score', 'revenue_at_risk', 'recommendation', 'risk_tier']
    ].copy()
    
    top_50 = top_50.reset_index(drop=True)
    top_50.index = top_50.index + 1
    top_50.index.name = 'Rank'
    
    top_50['churn_probability'] = top_50['churn_probability'].apply(lambda x: f"{x*100:.1f}%")
    top_50['monthly_revenue'] = top_50['monthly_revenue'].apply(lambda x: f"${x:,.0f}")
    top_50['revenue_at_risk'] = top_50['revenue_at_risk'].apply(lambda x: f"${x:,.0f}")
    top_50['health_score'] = top_50['health_score'].apply(lambda x: f"{x:.0f}")
    
    table = dbc.Table.from_dataframe(top_50, striped=True, bordered=True, hover=True, responsive=True, size='sm')
    return table


@app.callback(
    Output('revenue-at-risk-tier', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_revenue_tier(n):
    df = get_latest_predictions()
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available")
    
    revenue_by_tier = df.groupby('risk_tier')['annual_revenue_at_risk'].sum()
    colors = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
    
    fig = go.Figure(data=[
        go.Bar(
            x=revenue_by_tier.index,
            y=revenue_by_tier.values,
            marker=dict(color=[colors.get(tier, 'gray') for tier in revenue_by_tier.index])
        )
    ])
    
    fig.update_layout(
        title='Annual Revenue at Risk by Tier',
        xaxis_title='Risk Tier',
        yaxis_title='Revenue ($)',
        height=400
    )
    
    return fig


@app.callback(
    Output('churn-prob-scatter', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_scatter(n):
    df = get_latest_predictions()
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available")
    
    fig = px.scatter(
        df,
        x='health_score',
        y='churn_probability',
        color='risk_tier',
        size='monthly_revenue',
        hover_data=['customer_id', 'recommendation'],
        title='Churn Probability vs Health Score',
        color_discrete_map={
            'Low Risk': 'green',
            'Medium Risk': 'orange',
            'High Risk': 'red'
        }
    )
    
    fig.update_layout(height=400)
    return fig


@app.callback(
    Output('health-score-dist', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_health_dist(n):
    df = get_latest_predictions()
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available")
    
    fig = go.Figure(data=[
        go.Histogram(x=df['health_score'], nbinsx=20, marker=dict(color='skyblue'))
    ])
    
    fig.update_layout(
        title='Distribution of Health Scores',
        xaxis_title='Health Score (0-100)',
        yaxis_title='Number of Customers',
        height=400
    )
    
    return fig


@app.callback(
    Output('retention-scenarios', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_scenarios(n):
    df = get_latest_predictions()
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available")
    
    total_at_risk = df['annual_revenue_at_risk'].sum()
    
    scenarios = {
        '10%': total_at_risk * 0.10,
        '20%': total_at_risk * 0.20,
        '30%': total_at_risk * 0.30,
        '50%': total_at_risk * 0.50
    }
    
    fig = go.Figure(data=[
        go.Bar(x=list(scenarios.keys()), y=list(scenarios.values()), marker=dict(color='green'))
    ])
    
    fig.update_layout(
        title='Retention Scenario Analysis (Annual Revenue Saved)',
        xaxis_title='Retention Success Rate',
        yaxis_title='Annual Revenue Saved ($)',
        height=400
    )
    
    return fig


@app.callback(
    Output('business-metrics-text', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_business_text(n):
    df = get_latest_predictions()
    metrics = get_business_metrics()
    
    if df.empty:
        return dbc.Card([
            dbc.CardBody([
                dbc.Alert("No predictions available. Run the pipeline first.", color="warning")
            ])
        ])
    
    high_risk_df = df[df['risk_tier'] == 'High Risk']
    urgent_df = df[df['recommendation'] == 'Urgent Account Review']
    
    return dbc.Card([
        dbc.CardBody([
            html.H5("Business Case Summary", className="mb-3"),
            html.Hr(),
            html.P([html.B("Total Customers: "), f"{metrics['total_customers']:,}"], className="mb-2"),
            html.P([html.B("High Risk Customers: "), f"{metrics['high_risk']:,} ({metrics['high_risk']/metrics['total_customers']*100:.1f}%)"], className="mb-2"),
            html.P([html.B("Annual Revenue at Risk: "), f"${metrics['total_revenue_at_risk']:,.0f}"], className="mb-2"),
            html.P([html.B("Urgent Review Needed: "), f"{len(urgent_df):,} customers"], className="mb-3"),
            html.Hr(),
            html.H6("Retention ROI (20% Success Rate)"),
            html.P([html.B("Revenue Saved: "), f"${metrics['total_revenue_at_risk'] * 0.20:,.0f}"], className="mb-2"),
            html.P(["Focus on ", html.B("urgent account reviews"), " for highest impact."])
        ])
    ])


@app.callback(
    Output('recommendation-breakdown', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_rec_breakdown(n):
    df = get_latest_predictions()
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available")
    
    rec_revenue = df.groupby('recommendation')['monthly_revenue'].sum().sort_values(ascending=False)
    
    fig = go.Figure(data=[
        go.Bar(x=rec_revenue.index, y=rec_revenue.values, marker=dict(color='steelblue'))
    ])
    
    fig.update_layout(
        title='Monthly Revenue by Recommendation Type',
        xaxis_title='Recommendation',
        yaxis_title='Monthly Revenue ($)',
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig


# ============================================================================
# CALLBACKS - PRESCRIPTIONS & JOURNEY ANALYSIS
# ============================================================================

@app.callback(
    Output('customer-dropdown', 'options'),
    [Input('interval-component', 'n_intervals')]
)
def update_customer_dropdown(n):
    """Populate customer dropdown."""
    journeys = get_journeys()
    if journeys.empty:
        return []
    return [{'label': f"Customer {cid}", 'value': cid} for cid in journeys['customer_id'].sort_values()]


@app.callback(
    Output('customer-dropdown', 'value'),
    [Input('customer-dropdown', 'options')]
)
def set_customer_dropdown_value(options):
    """Set default customer in dropdown."""
    if options:
        return options[0]['value']
    return None


@app.callback(
    Output('prescription-metrics', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_prescription_metrics(n):
    """Show portfolio-level prescription metrics."""
    prescriptions = get_prescriptions()
    
    if prescriptions.empty:
        return dbc.Alert("No prescriptions available. Run src/prescriptions.py first.", color="warning")
    
    total_cost = prescriptions['action_cost'].sum()
    total_impact = prescriptions['expected_impact'].sum()
    portfolio_roi = total_impact / total_cost if total_cost > 0 else 0
    high_roi_actions = len(prescriptions[prescriptions['roi'] > 10])
    
    return html.Div([
        html.P([html.B("Total Intervention Cost: "), f"${total_cost:,.0f}"], className="mb-2"),
        html.P([html.B("Total Expected Impact: "), f"${total_impact:,.0f}"], className="mb-2"),
        html.P([html.B("Portfolio ROI: "), f"{portfolio_roi:.1f}x"], className="mb-2", style={'color': 'green', 'font-size': '1.2em'}),
        html.P([html.B("High ROI Actions: "), f"{high_roi_actions:,} customers"], className="mb-0")
    ])


@app.callback(
    Output('roi-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_roi_chart(n):
    """Top 20 customers by action ROI bar chart."""
    prescriptions = get_prescriptions()
    if prescriptions.empty:
        return go.Figure().add_annotation(text="No prescriptions available")
    
    top_roi = prescriptions.nlargest(20, 'roi')
    fig = go.Figure(data=[
        go.Bar(x=top_roi['customer_id'].astype(str), y=top_roi['roi'], marker=dict(color=top_roi['roi'], colorscale='Viridis', showscale=True))
    ])
    fig.update_layout(title='Top 20 Customers by Action ROI', xaxis_title='Customer ID', yaxis_title='ROI', height=400)
    return fig


@app.callback(
    Output('action-type-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_action_type_chart(n):
    """Customers by action type bar chart."""
    prescriptions = get_prescriptions()
    if prescriptions.empty:
        return go.Figure().add_annotation(text="No prescriptions available")
    
    action_summary = prescriptions.groupby('action_type').agg({'customer_id': 'count'}).reset_index()
    action_summary.columns = ['Action Type', 'Count']
    
    fig = go.Figure(data=[go.Bar(x=action_summary['Action Type'], y=action_summary['Count'], marker=dict(color='steelblue'))])
    fig.update_layout(title='Customers by Action Type', xaxis_title='Action Type', yaxis_title='Count', height=400)
    return fig


@app.callback(
    Output('prescriptions-table', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_prescriptions_table(n):
    """Top 20 prescriptions table."""
    prescriptions = get_prescriptions()
    if prescriptions.empty:
        return dbc.Alert("No prescriptions available.", color="warning")
    
    top_20 = prescriptions.head(20)[['customer_id', 'prescribed_action', 'discount_percent', 'roi', 'confidence', 'success_probability']].copy()
    top_20 = top_20.reset_index(drop=True)
    top_20.index = top_20.index + 1
    top_20.index.name = 'Rank'
    
    top_20['discount_percent'] = top_20['discount_percent'].apply(lambda x: f"{x:.0f}%")
    top_20['roi'] = top_20['roi'].apply(lambda x: f"{x:.1f}x")
    top_20['confidence'] = top_20['confidence'].apply(lambda x: f"{x*100:.0f}%")
    top_20['success_probability'] = top_20['success_probability'].apply(lambda x: f"{x*100:.0f}%")
    
    table = dbc.Table.from_dataframe(top_20, striped=True, bordered=True, hover=True, responsive=True, size='sm')
    return table


@app.callback(
    [Output('journey-summary', 'children'), Output('journey-chart', 'figure'), Output('journey-table', 'children')],
    [Input('customer-dropdown', 'value')]
)
def update_journey(selected_customer):
    """Update journey summary, chart, and table."""
    if not selected_customer:
        return (dbc.Alert("Select a customer", color="info"), go.Figure().add_annotation(text="Select a customer"), dbc.Alert("Select a customer", color="info"))
    
    journey_df = get_journey_details(selected_customer)
    if journey_df.empty:
        return (dbc.Alert(f"No data for customer {selected_customer}", color="warning"), go.Figure().add_annotation(text="No data"), dbc.Alert(f"No data", color="warning"))
    
    current = journey_df.iloc[-1]
    summary = html.Div([
        html.P([html.B("Logins/Day: "), f"{current['logins_per_day']:.2f}"], className="mb-2"),
        html.P([html.B("Days Inactive: "), f"{int(current['days_since_last_login'])}"], className="mb-2"),
        html.P([html.B("Features Used: "), f"{int(current['features_used_count'])} / 10"], className="mb-2"),
        html.P([html.B("Engagement Score: "), f"{current['engagement_score']:.0f}"], className="mb-0")
    ])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=journey_df['year_month'], y=journey_df['logins_per_day'], mode='lines+markers', name='Logins/Day', yaxis='y1'))
    fig.add_trace(go.Scatter(x=journey_df['year_month'], y=journey_df['engagement_score'], mode='lines+markers', name='Engagement Score', yaxis='y2'))
    fig.update_layout(title=f"Customer {selected_customer} Journey", xaxis_title='Month', yaxis=dict(title='Logins/Day'), yaxis2=dict(title='Engagement Score', overlaying='y', side='right'), height=400, hovermode='x unified')
    
    table_df = journey_df[['year_month', 'logins_per_day', 'features_used_count', 'engagement_score', 'days_since_last_login']].copy()
    table_df.columns = ['Month', 'Logins/Day', 'Features Used', 'Engagement', 'Days Inactive']
    table_df = table_df.reset_index(drop=True)
    table_df.index = table_df.index + 1
    table_df.index.name = 'Period'
    
    table = dbc.Table.from_dataframe(table_df, striped=True, bordered=True, hover=True, responsive=True, size='sm')
    return summary, fig, table


@app.callback(
    [Output('phase-distribution-chart', 'figure'), Output('risk-timeline-chart', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_journey_distributions(n):
    """Journey phase and risk distribution charts."""
    journeys = get_journeys()
    if journeys.empty:
        return (go.Figure().add_annotation(text="No data"), go.Figure().add_annotation(text="No data"))
    
    phase_dist = journeys['current_phase'].value_counts()
    fig1 = go.Figure(data=[go.Bar(x=phase_dist.index, y=phase_dist.values, marker=dict(color='steelblue'))])
    fig1.update_layout(title='Current Phase Distribution', xaxis_title='Phase', yaxis_title='Count', height=400)
    
    risk_dist = journeys['current_risk'].value_counts()
    colors_risk = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green', 'CHURNED': 'darkred'}
    fig2 = go.Figure(data=[go.Bar(x=risk_dist.index, y=risk_dist.values, marker=dict(color=[colors_risk.get(r, 'gray') for r in risk_dist.index]))])
    fig2.update_layout(title='Current Risk Distribution', xaxis_title='Risk Level', yaxis_title='Count', height=400)
    
    return fig1, fig2


# ============================================================================
# CALLBACKS - SHAP EXPLAINABILITY
# ============================================================================

@app.callback(
    Output('shap-customer-dropdown', 'options'),
    [Input('interval-component', 'n_intervals')]
)
def update_shap_customer_dropdown(n):
    """Populate SHAP customer dropdown."""
    shap_df = get_shap_explanations()
    if shap_df.empty:
        return []
    
    predictions = get_latest_predictions()
    return [{'label': f"Customer {cid}", 'value': cid} for cid in predictions['customer_id'].sort_values()]


@app.callback(
    Output('shap-customer-dropdown', 'value'),
    [Input('shap-customer-dropdown', 'options')]
)
def set_shap_customer_dropdown_value(options):
    """Set default customer in SHAP dropdown."""
    if options:
        return options[0]['value']
    return None


@app.callback(
    [Output('shap-summary', 'children'), Output('shap-explanation-table', 'children')],
    [Input('shap-customer-dropdown', 'value')]
)
def update_shap_explanation(selected_customer):
    """Update SHAP explanation for selected customer."""
    if not selected_customer:
        return (
            dbc.Alert("Select a customer", color="info"),
            dbc.Alert("Select a customer", color="info")
        )
    
    predictions = get_latest_predictions()
    customer_pred = predictions[predictions['customer_id'] == selected_customer]
    
    if customer_pred.empty:
        return (
            dbc.Alert(f"No data for customer {selected_customer}", color="warning"),
            dbc.Alert(f"No data", color="warning")
        )
    
    pred = customer_pred.iloc[0]
    churn_prob = pred['churn_probability']
    health_score = pred['health_score']
    recommendation = pred['recommendation']
    
    # Summary card
    summary = html.Div([
        html.P([html.B("Predicted Churn Probability: "), f"{churn_prob*100:.1f}%"], className="mb-2"),
        html.P([html.B("Health Score: "), f"{health_score:.0f}/100"], className="mb-2"),
        html.P([html.B("Recommendation: "), recommendation], className="mb-2"),
        html.Hr(),
        html.P([
            "The features below show which factors contributed most to this churn prediction. ",
            "Positive values increase churn risk, negative values decrease it."
        ], style={'font-size': '0.9em', 'color': '#666'})
    ])
    
    # Get SHAP explanations
    shap_df = get_shap_explanations()
    
    if shap_df.empty:
        table = dbc.Alert(
            "No SHAP explanations available. Run src/shap_analysis.py first.",
            color="warning"
        )
    else:
        customer_shap = shap_df[shap_df['customer_id'] == selected_customer]
        
        if customer_shap.empty:
            table = dbc.Alert(f"No SHAP data for customer {selected_customer}", color="warning")
        else:
            row = customer_shap.iloc[0]
            
            # Parse top 3 features
            features = row['top_3_features'].split(', ')
            values = [float(v) for v in row['top_3_shap_values'].split(', ')]
            
            # Create table
            table_data = []
            for i, (feat, val) in enumerate(zip(features, values)):
                direction = "↑ Increases Risk" if val > 0 else "↓ Decreases Risk"
                color = '#ff6b6b' if val > 0 else '#51cf66'
                
                table_data.append({
                    'Rank': i + 1,
                    'Feature': feat,
                    'Impact Value': f"{val:.4f}",
                    'Direction': direction
                })
            
            table_df = pd.DataFrame(table_data)
            
            table = html.Div([
                dbc.Table.from_dataframe(
                    table_df,
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True,
                    size='sm'
                ),
                html.Hr(),
                html.P([
                    html.B("How to interpret: "),
                    "These are the 3 most important features affecting this customer's churn prediction. "
                    "Red/up arrows = increases churn risk. Green/down arrows = decreases churn risk."
                ], style={'font-size': '0.9em', 'color': '#666', 'margin-top': '10px'})
            ])
    
    return summary, table


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("DASH DASHBOARD STARTING")
    print("=" * 70)
    print("\nDashboard running at: http://127.0.0.1:8050")
    print("Open your browser and go to: http://127.0.0.1:8050")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(debug=True, port=8050)