import numpy as np
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

BG_COLOR, SURFACE_COLOR, BORDER_COLOR = "#0d0d0d", "#1c1c1e", "#3a3a3c"
TEXT_PRIMARY, TEXT_SECONDARY = "#ffffff", "#aeaeb2"
COLORS = {'critical': '#ff453a', 'poor': '#ff9f0a', 'fair': '#ffd60a', 'good': '#30d158', 'excellent': '#0a84ff'}

def get_f_color(f_val: float) -> str:
    if f_val < 0.40: return COLORS['critical']
    if f_val < 0.60: return COLORS['poor']
    if f_val < 0.70: return COLORS['fair']
    if f_val < 0.90: return COLORS['good']
    return COLORS['excellent']

def create_gauge(val: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val, title={'text': title, 'font': {'color': TEXT_SECONDARY, 'size': 14}},
        number={'font': {'color': TEXT_PRIMARY, 'size': 24, 'family': 'sans-serif'}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': BORDER_COLOR},
            'bar': {'color': get_f_color(val)},
            'bgcolor': SURFACE_COLOR,
            'borderwidth': 2, 'bordercolor': BORDER_COLOR,
            'steps': [
                {'range': [0, 0.4], 'color': 'rgba(255, 69, 58, 0.1)'},
                {'range': [0.4, 0.7], 'color': 'rgba(255, 159, 10, 0.1)'},
                {'range': [0.7, 1.0], 'color': 'rgba(10, 132, 255, 0.1)'}],
        }))
    fig.update_layout(paper_bgcolor=BG_COLOR, font={'color': TEXT_PRIMARY, 'family': 'sans-serif'},
                      margin=dict(l=10, r=10, t=30, b=10), height=150)
    return fig

def create_app() -> dash.Dash:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
    app.layout = dbc.Container([
        dcc.Interval(id='interval-component', interval=2000, n_intervals=0),
        dbc.Row([
            dbc.Col(html.H2("PLANTAOS · HORSE CFT", style={'color': TEXT_PRIMARY, 'fontWeight': 'bold'}), width=8),
            dbc.Col(html.H2(id='f-global-display', style={'color': COLORS['excellent'], 'textAlign': 'right', 'fontWeight': 'bold'}), width=4)
        ], className="mt-3 pb-2 border-bottom border-secondary"),
        dbc.Row([
            dbc.Col(dcc.Graph(id='gauge-thermal', config={'displayModeBar': False}), width=2),
            dbc.Col(dcc.Graph(id='gauge-air', config={'displayModeBar': False}), width=2),
            dbc.Col(dcc.Graph(id='gauge-energy', config={'displayModeBar': False}), width=2),
            dbc.Col(dcc.Graph(id='gauge-light', config={'displayModeBar': False}), width=2),
            dbc.Col(dcc.Graph(id='gauge-occ', config={'displayModeBar': False}), width=2),
            dbc.Col(html.Div(id='regime-badge', className="mt-4 text-center"), width=2)
        ], className="mt-3")
    ], fluid=True, style={'backgroundColor': BG_COLOR, 'minHeight': '100vh'})

    @app.callback(
        [Output('f-global-display', 'children'), Output('gauge-thermal', 'figure'),
         Output('gauge-air', 'figure'), Output('gauge-energy', 'figure'),
         Output('gauge-light', 'figure'), Output('gauge-occ', 'figure'),
         Output('regime-badge', 'children')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_dashboard(n):
        np.random.seed(n)
        f_t, f_a, f_e, f_l, f_o = np.clip(np.random.normal(0.8, 0.1, 5), 0.01, 1.0)
        f_g = 5.0 / (1/f_t + 1/f_a + 1/f_e + 1/f_l + 1/f_o)
        
        regime = "INTELLIGENT" if f_g > 0.75 else "ACTIVE"
        badge = html.H4(regime, style={'color': SURFACE_COLOR, 'backgroundColor': get_f_color(f_g), 'padding': '10px', 'borderRadius': '5px'})
        
        return [f"F_global: {f_g:.2f}", create_gauge(f_t, "F_thermal"), create_gauge(f_a, "F_air"),
                create_gauge(f_e, "F_energy"), create_gauge(f_l, "F_light"), create_gauge(f_o, "F_occupancy"), badge]
    
    return app
