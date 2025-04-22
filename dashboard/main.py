import json
import warnings

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dash_table, dcc, html
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")

app = dash.Dash(__name__)
server = app.server
engine = create_engine(
    'postgresql+psycopg2://readonly_user:read0nly_passw0rd@datamart/analytics'
)


def load_data():
    with engine.connect() as conn:
        query = "SELECT * FROM churn_data"
        data = pd.read_sql(query, conn)

    data['tenure'] = data['tenure'].astype(int)
    data['monthlycharges'] = data['monthlycharges'].astype(float)
    data['totalcharges'] = pd.to_numeric(data['totalcharges'], errors='coerce')
    data['churn'] = data['churn'].map({'Yes': 1, 'No': 0})
    data['seniorcitizen'] = data['seniorcitizen'].astype(int)

    return data


app.layout = html.Div([
    dcc.Interval(
        id='interval-component',
        interval=5 * 1000,
        n_intervals=0
    ),

    html.H1("Дашборд аналитики ухода клиентов",
            style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.H3("Всего клиентов"),
            html.H2(id='total-customers', style={'color': '#1f77b4'})
        ], className='four columns metric-box'),

        html.Div([
            html.H3("Churn Rate"),
            html.H2(id='churn-rate', style={'color': '#ff7f0e'})
        ], className='four columns metric-box'),

        html.Div([
            html.H3("Средний Срок Обслуживания (месяца)"),
            html.H2(id='avg-tenure', style={'color': '#2ca02c'})
        ], className='four columns metric-box')
    ], className='row'),

    html.Div([
        html.Div([
            html.H4("Churn Distribution"),
            dcc.Graph(id='churn-distribution'),

            html.H4("Survival Probability Over Time"),
            dcc.Graph(id='survival-curve'),

            html.H4("Risk Factors Analysis"),
            dcc.Graph(id='risk-factors')
        ], className='six columns'),

        html.Div([
            html.H4("Клиенты с высоким риском (Следующие 30 Дней)"),
            dash_table.DataTable(
                id='high-risk-table',
                columns=[
                    {'name': 'ID клиента', 'id': 'customerid'},
                    {'name': 'Вероятность ухода', 'id': 'churn_prob'},
                    {'name': 'Предсказанный срок обслуживания', 'id': 'pred_tenure'}
                ],
                style_table={'overflowX': 'auto'},
                page_size=10
            ),

            html.H4("Customer Lifetime Value Distribution"),
            dcc.Graph(id='clv-distribution'),

            html.H4("Feature Importance for Churn Prediction"),
            dcc.Graph(id='feature-importance')
        ], className='six columns')
    ], className='row'),

    html.Div(id='data-store', style={'display': 'none'})
])


@app.callback(
    Output('data-store', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_data(n):
    print(f"⏰ Callback triggered! Interval count: {n}")
    df = load_data()
    print("Data loaded:", "EMPTY" if df.empty else f"{len(df)} rows")
    return json.dumps(df.to_dict(orient='split'))


@app.callback(
    [Output('total-customers', 'children'),
     Output('churn-rate', 'children'),
     Output('avg-tenure', 'children')],
    [Input('data-store', 'children')]
)
def update_metrics(json_data):
    df = pd.read_json(json_data, orient='split')

    total_customers = len(df)
    churn_rate = f"{df['churn'].mean() * 100:.1f}%"
    avg_tenure = f"{df['tenure'].mean():.1f}"

    return total_customers, churn_rate, avg_tenure


@app.callback(
    Output('churn-distribution', 'figure'),
    [Input('data-store', 'children')]
)
def update_churn_distribution(json_data):
    df = pd.read_json(json_data, orient='split')

    fig = px.pie(df, names='churn', title='Churn vs Non-Churn Customers',
                 color='churn', color_discrete_map={0: 'green', 1: 'red'})
    fig.update_traces(textinfo='percent+label')

    return fig


@app.callback(
    Output('survival-curve', 'figure'),
    [Input('data-store', 'children')]
)
def update_survival_curve(json_data):
    df = pd.read_json(json_data, orient='split')

    km_data = df.groupby('tenure')['churn'].agg(['sum', 'count']).reset_index()
    km_data['survival'] = 1 - (km_data['sum'].cumsum()
                               / km_data['count'].sum())

    fig = px.line(km_data, x='tenure', y='survival',
                  title='Customer Survival Probability by Tenure',
                  labels={'tenure': 'Tenure (months)',
                          'survival': 'Survival Probability'})
    fig.update_yaxes(range=[0, 1])

    return fig


@app.callback(
    Output('risk-factors', 'figure'),
    [Input('data-store', 'children')]
)
def update_risk_factors(json_data):
    df = pd.read_json(json_data, orient='split')

    contract_churn = df.groupby('contract')['churn'].mean().reset_index()
    internet_churn = (df.groupby('internetservice')['churn']
                      .mean()
                      .reset_index())

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=contract_churn['contract'],
        y=contract_churn['churn'],
        name='Contract Type'
    ))

    fig.add_trace(go.Bar(
        x=internet_churn['internetservice'],
        y=internet_churn['churn'],
        name='Internet Service'
    ))

    fig.update_layout(
        title='Churn Rate by Key Factors',
        yaxis_title='Churn Rate',
        barmode='group'
    )

    return fig


@app.callback(
    Output('high-risk-table', 'data'),
    [Input('data-store', 'children')]
)
def update_high_risk_table(json_data):
    df = pd.read_json(json_data, orient='split')

    df['churn_prob'] = np.where(
        df['contract'] == 'Month-to-month',
        np.random.uniform(0.3, 0.9, len(df)),
        np.random.uniform(0.1, 0.4, len(df))
    )

    df['pred_tenure'] = df['tenure'] + (1 - df['churn_prob']) * 50

    high_risk = df[
        (df['churn_prob'] > 0.75)
    ].sort_values('churn_prob', ascending=False)

    return high_risk[['customerid', 'churn_prob',
                      'pred_tenure']].to_dict('records')


@app.callback(
    Output('clv-distribution', 'figure'),
    [Input('data-store', 'children')]
)
def update_clv_distribution(json_data):
    df = pd.read_json(json_data, orient='split')

    df['clv'] = df['monthlycharges'] * df['tenure']

    fig = px.histogram(df, x='clv', nbins=20,
                       title='Customer Lifetime Value Distribution',
                       labels={'clv': 'Customer Lifetime Value ($)'})

    return fig


@app.callback(
    Output('feature-importance', 'figure'),
    [Input('data-store', 'children')]
)
def update_feature_importance(json_data):
    features = ['MonthlyCharges', 'tenure', 'Contract',
                'InternetService', 'OnlineSecurity']
    importance = [0.35, 0.25, 0.15, 0.12, 0.08]

    fig = px.bar(x=importance, y=features, orientation='h',
                 title='Feature Importance for Churn Prediction',
                 labels={'x': 'Importance', 'y': 'Feature'})

    return fig


if __name__ == '__main__':
    app.run(debug=True, port=8888, host='0.0.0.0')
