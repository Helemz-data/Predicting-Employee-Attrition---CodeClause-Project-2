import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import os

MODEL_PATH = "artifacts/attrition_rf_model.pkl"
FEATURES_PATH = "artifacts/feature_columns.pkl"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

DATA_PATH = "HR Employee Attrition.csv"
df = pd.read_csv(DATA_PATH)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H2("HR Attrition Dashboard"), className="mb-4")]),
    dbc.Row([
        dbc.Col(dbc.Card([dcc.Graph(id='fig-1', figure={})], body=True), md=6),
        dbc.Col(dbc.Card([dcc.Graph(id='fig-2', figure={})], body=True), md=6),
    ]),
    dbc.Row([dbc.Col(html.H4("Predict Attrition for a single employee"))]),
    dbc.Row([
        dbc.Col(dbc.Input(id='input-age', placeholder='Age', type='number', value=30), md=2),
        dbc.Col(dcc.Dropdown(id='input-jobrole',
                             options=[{'label': r, 'value': r} for r in sorted(df['JobRole'].unique())],
                             value=df['JobRole'].unique()[0]), md=4),
        dbc.Col(dcc.Dropdown(id='input-over18',
                             options=[{'label': 'No', 'value': 'No'},
                                      {'label': 'Yes', 'value': 'Yes'}],
                             value='Yes'), md=2),
        dbc.Col(dbc.Button('Predict', id='predict-btn', color='primary'), md=2)
    ], className='mb-4'),
    dbc.Row([dbc.Col(html.Div(id='prediction-output'))])
], fluid=True)

@app.callback(
    Output('fig-1', 'figure'),
    Output('fig-2', 'figure'),
    Input('predict-btn', 'n_clicks')
)
def update_figs(n):
    fig1 = px.histogram(df, x='Age', nbins=20, title='Age distribution')
    fig2 = px.sunburst(df, path=['BusinessTravel', 'Department', 'Attrition'],
                       title='Attrition by dept & travel')
    return fig1, fig2

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('input-age', 'value'),
    State('input-jobrole', 'value'),
    State('input-over18', 'value')
)
def predict(n, age, jobrole, over18):
    if n is None:
        return ''
    row = pd.Series(0, index=feature_columns)
    if 'Age' in row.index:
        row['Age'] = (age - df['Age'].mean()) / df['Age'].std()
    colname = f'JobRole_{jobrole}'
    if colname in row.index:
        row[colname] = 1
    X = row.values.reshape(1, -1)
    pred_proba = model.predict_proba(X)[0, 1]
    return html.Div([
        html.P(f'Probability of leaving: {pred_proba:.2f}'),
        html.P('Recommendation: higher than 0.5 → elevated risk')
    ])

if __name__ == "__main__":
    app.run(debug=True, port=8050)
