##########################################################
# Amalia Jiménez Toledano and Roberto Jesús Alcaraz Molina
##########################################################

import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_table as dt
import pandas as pd
import plotly.express as px
import json


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
df1 = pd.read_csv('https://raw.githubusercontent.com/RoberAlcaraz/First-Take-Away/main/vehicles_data.csv')

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Final Project", className="display-4"),
        html.H4("Data Tidying and reporting", className="display-6"),
        html.Hr(),
        html.P(
            "We are going to analyze two different data sets", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Introduction", href="/", active="exact"),
                dbc.NavLink("Data 1", href="/page-1", active="exact"),
                dbc.NavLink("Data 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

###################################################################
# LAYOUT
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])

###################################################################
# CALLBACK
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return [
                html.H1('Introduction',
                        style={'textAlign':'center'})
                ]
    elif pathname == "/page-1":
        return [
                html.H1('Data 1',
                        style={'textAlign':'center'})
                ]
    elif pathname == "/page-2":
        return [
                html.H1('Data 2',
                        style={'textAlign':'center'})
                ]
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__=='__main__':
    app.run_server()


