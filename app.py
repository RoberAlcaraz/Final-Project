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

df1_cont = df1[['year', 'odometer']]
df1_cat = df1[['price', 'manufacturer', 'condition', 'fuel', 'title_status', 'transmission', 'drive']]


## Bank Churnes Data Set

df2 = pd.read_csv('https://raw.githubusercontent.com/amaliajimenezajt/final_shiny_app/master/BankChurnersData.csv')
df2["Attrition_Flag"] = df2["Attrition_Flag"].astype("category")
df2["Gender"] = df2["Gender"].astype("category")
df2["Education_Level"] = df2["Education_Level"].astype("category")
df2["Education_Level"] = df2["Education_Level"].astype("category")
df2["Card_Category"] = df2["Card_Category"].astype("category")
df2["Income_Category_final"] = df2["Income_Category_final"].astype("category")


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
                 dbc.NavLink("Bank Churners: Data description", href="/page-4", active="exact"),
                 dbc.NavLink("Bank Churners: Variables Plots", href="/page-5", active="exact"),
                 dbc.NavLink("Bank Churners: Regression Plot-Summary", href="/page-6", active="exact")
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
                html.H1('Introduction', style={'textAlign':'center'}),
                html.P('Craigslist is the world’s largest collection of used vehicles for sale.'
                'This data set includes every used vehicle entry within the United States on'
                'Craiglist, from the year 1900 until today. This data set has been taken'
                'from the website'),
                html.P('A bank manager is interested in predicting the annual income of his or her clients account holder.'
                'For the new year, the bank has decided to create a new service depending on this income,'
                'so that it will be able to know which customers have good income in order to give'
                'them a better service and make them commit to stay with the bank.')
                ]

    elif pathname == "/page-1":
        return [
        html.H1('Data description', style={'textAlign':'center'}),
        html.Br(),
        html.Div(id='my-div'),
        html.Br(),
        dt.DataTable(
            id='datatable-interactivity1',
            columns=[
                {"name": i, "id": i, "deletable": True, "selectable": True} for i in df1.columns
            ],
            data=df1.to_dict('records'),
            filter_action="native",
            sort_action="native"
        ),
        html.Div(id='datatable-interactivity-container1')
                ]
    elif pathname == "/page-2":
        return [
        html.H1('Descriptive analysis'),
        dcc.Dropdown(
            id='cont-vars',
            options=[{'label': i, 'value': i} for i in df1_cont.columns]
            ),
        dcc.Graph(id='hist'),
        dcc.Dropdown(
            id='cat-vars',
            options=[{'label': i, 'value': i} for i in df1_cat.columns]
            ),
        dcc.Graph(id='bar')
        ]
    elif pathname == "/page-3":
        return []
    elif pathname == "/page-4":
        return [
                html.H1('Data Description-Summary',
                        style={'textAlign':'center'}),
                html.Div(
                    [dt.DataTable(
                        id='datatable-interactivity',
                        columns=[
                            {"name": i, "id": i, "deletable": True, "selectable": True} for i in df2.columns
                            ],
                            data=df2.to_dict('records'),
                            editable=True,
                            filter_action="native",
                            sort_action="native",
                            sort_mode="multi",
                            column_selectable="single",
                            row_selectable="multi",
                            row_deletable=True,
                            selected_columns=[],
                            selected_rows=[],
                            page_action="native",
                            page_current= 0,
                            page_size= 10,
                            ),
                html.Div(id='datatable-interactivity-container')
                ])
                ]
    elif pathname == "/page-5":
        return [
        html.H1('Plot numerical vs categorical',
        style={'textAlign':'center'}),
        html.P('In the following graph, you can select the numerical variable according to' 
        'the most relevant categorical variables. Moreover, as the response variable in this case' 
        'is income, it is important to have this reference as well:',
        style={'textAlign':'center',
        'color':'green'}),
        html.P("Categorical Variable:",
        style={'color':'red'}),
        dcc.RadioItems(
        id='x-axis', 
        options=[{'value': x, 'label': x} 
                 for x in ['Attrition_Flag', 'Education_Level',  'Card_Category']],
        value=['Education_Level'], 
        labelStyle={'display': 'inline-block'}
        ),
        html.P("Numerical Variable:",
        style={'color':'blue'}),
    dcc.RadioItems(
        id='y-axis', 
        options=[{'value': x, 'label': x} 
                 for x in ['Customer_Age', 'Total_Revolving_Bal', 'Total_Trans_Amt', 'Total_Trans_Ct']],
        value='Total_Trans_Ct', 
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Graph(id="box-plot")
    ]
    
    elif pathname == "/page-6":
        return [
        html.P("Categorical Variable:",
        style={'color':'red'}),
        dcc.RadioItems(
        id='x-linear', 
        options=[{'value': x, 'label': x} 
                 for x in ['Customer_Age', 'Total_Revolving_Bal', 'Total_Trans_Amt', 'Total_Trans_Ct']],
        value=['Customer_Age'], 
        labelStyle={'display': 'inline-block'}),
        dcc.RadioItems(
        id='y-linear', 
        options=[{'value': x, 'label': x} 
                 for x in ['Customer_Age', 'Total_Revolving_Bal', 'Total_Trans_Amt', 'Total_Trans_Ct']],
        value='Total_Trans_Ct', 
        labelStyle={'display': 'inline-block'}),
        
        dcc.Graph(id="linear")
        
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
    Output('datatable-interactivity1', 'style_data_conditional1'),
    Input('datatable-interactivity1', 'selected_columns1')
)
def update_styles(selected_columns1):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns1]

@app.callback(
    Output('hist', 'figure'),
    Input('cont-vars', 'value'))
def update_hist(selected_var):
    fig = px.histogram(df1, x=selected_var)
    return fig

@app.callback(
    Output('bar', 'figure'),
    Input('cat-vars', 'value'))
def update_bar(selected_var):
    fig = px.bar(df1, x=selected_var, color='price')
    return fig


### table df2

@app.callback(
    Output('datatable-interactivity', 'style_data_conditional'),
    Input('datatable-interactivity', 'selected_columns')
)

def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]

@app.callback(
    Output('datatable-interactivity-container', "children"),
    Input('datatable-interactivity', "derived_virtual_data"),
    Input('datatable-interactivity', "derived_virtual_selected_rows"))
    
def update_graphs(rows, derived_virtual_selected_rows):

    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    dff = df2 if rows is None else pd.DataFrame(rows)

    colors = ['#7FDBFF' if i in derived_virtual_selected_rows else '#0074D9'
              for i in range(len(dff))]

    return [
        dcc.Graph(
            id=column,
            figure={
                "data": [
                    {
                        "x": dff["Education_Level"],
                        "y": dff[column],
                        "type": "bar",
                        "marker": {"color": colors},
                    }
                ],
                "layout": {
                    "xaxis": {"automargin": True},
                    "yaxis": {
                        "automargin": True,
                        "title": {"text": column}
                    },
                    "height": 250,
                    "margin": {"t": 10, "l": 10, "r": 10},
                },
            },
        )
        # check if column exists - user may have deleted it
        # If `column.deletable=False`, then you don't
        # need to do this check.
        for column in ["Total_Revolving_Bal", "Total_Trans_Amt", "Total_Trans_Ct"] if column in dff
    ]



######## box-plot

@app.callback(
    Output("box-plot", "figure"), 
    [Input("x-axis", "value"), 
     Input("y-axis", "value")])
def generate_chart(x, y):
    fig = px.box(df2, x=x, y=y,color='Income_Category_final',
    title="Box plot of Numerical Variables",
    notched=True)
    return fig

######### linear regression
@app.callback(
    Output("linear", "figure"), 
    [Input("x-linear", "value"),
    Input("y-linear", "value")])
def generate_linear(x,y):
    fig = px.scatter(df2, x=x, y=y, facet_col="Attrition_Flag", color="Income_Category_final", trendline="ols")
    results = px.get_trendline_results(fig)
    return fig
    print(results)
    results.query("sex == 'Male' and smoker == 'Yes'").px_fit_results.iloc[0].summary()


    








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




