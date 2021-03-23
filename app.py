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

# Vehicles data set
df1 = pd.read_csv('https://raw.githubusercontent.com/RoberAlcaraz/First-Take-Away/main/vehicles_data.csv')
df1["price"] = df1["price"].astype("category")
df1["manufacturer"] = df1["manufacturer"].astype("category")
df1["condition"] = df1["condition"].astype("category")
df1["fuel"] = df1["fuel"].astype("category")
df1["title_status"] = df1["title_status"].astype("category")
df1["transmission"] = df1["transmission"].astype("category")
df1["drive"] = df1["drive"].astype("category")
df1 = df1.drop(columns=['state', 'lat', 'long'])

# Bank Churnes Data Set

df2 = pd.read_csv('https://raw.githubusercontent.com/amaliajimenezajt/final_shiny_app/master/BankChurnersData.csv')
df2["Attrition_Flag"] = df2["Attrition_Flag"].astype("category")
df2["Gender"] = df2["Gender"].astype("category")
df2["Education_Level"] = df2["Education_Level"].astype("category")
df2["Education_Level"] = df2["Education_Level"].astype("category")
df2["Card_Category"] = df2["Card_Category"].astype("category")
df2["Income_Category_final"] = df2["Income_Category_final"].astype("category")



###############################################################################
# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "22rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "24rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Final Project"),
        html.H4("Data Tidying and reporting"),
        html.Hr(),
        html.H6("Amalia Jiménez Toledano and Roberto Jesús Alcaraz Molina"),
        html.Hr(),
        html.P(
            "We are going to analyze two different data sets", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Introduction", href="/", active="exact"),
                dbc.NavLink("Vehicles: Data description", href="/page-1", active="exact"),
                dbc.NavLink("Vehicles: Descriptive analysis", href="/page-2", active="exact"),
                dbc.NavLink("Vehicles: Statistical models", href="/page-3", active="exact"),
                dbc.NavLink("Data 2", href="/page-4", active="exact"),
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

# introPage = [
#     html.H1('Introduction', style={'textAlign':'center'}),
#     html.P('Craigslist is the world’s largest collection of used vehicles for sale.'
#     'This data set includes every used vehicle entry within the United States on'
#     'Craiglist, from the year 1900 until today. This data set has been taken'
#     'from the website')
#     ]
    
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return [
                html.H1('Introduction', style={'textAlign':'center'}),
                html.P('Craigslist is the world’s largest collection of used vehicles for sale.'
                'This data set includes every used vehicle entry within the United States on'
                'Craiglist, from the year 1900 until today. This data set has been taken'
                'from the website')
                ]
                
    elif pathname == "/page-1":
        return [
                html.H1('Data description', style={'textAlign':'center'}),
                html.Div(["Input: ",
                          dcc.Input(id='my-input', value='initial value', type='text')]),
                html.Br(),
                html.Div(id='my-output'),
                dcc.Dropdown(
                    id='my-vars',
                    options=[{'label': i, 'value': i} for i in df1.columns],
                    value=list(df1.columns.values)[0]
                    ),
                html.Br(),
                html.Div(id='my-div'),
                html.Br(),
                dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in df1.columns],
                    data=df1.to_dict('records'),
                )
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
    
@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    return 'Output: {}'.format(input_value)

# @app.callback(
#     Output('my-div', 'info'),
#     Input('my-vars', 'value')
# )
# def info_var(input_var):
#     filtered_df = df1[input_var]
#     df = pd.DataFrame(filtered_df)
#     return df.describe()

if __name__=='__main__':
    app.run_server()
