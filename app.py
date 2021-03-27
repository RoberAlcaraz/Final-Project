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
import statsmodels.api as sm
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
            sort_action="native",
            page_size= 10
        ),
        html.Div(id='datatable-interactivity-container1')
                ]
    elif pathname == "/page-2":
        return [
        html.H1('Descriptive analysis'),
        dcc.Tabs(id = "tabs", value = "tab-cont", children=[
            dcc.Tab(label = "Continuous variables", value="tab-cont"),
            dcc.Tab(label = "Categorical variables", value="tab-cat")
        ]),
        html.Div(id = "tabs-content")
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
                    page_size= 10,
                    filter_action="native",
                    sort_action="native",
                    )
        ])
        ]

    elif pathname == "/page-5":
        return [
        html.H1('Plot numerical vs categorical',
        style={'textAlign':'center'}),
        dcc.Tabs([
          dcc.Tab(label="Numerical Variables",children=[
          html.P('In the following graph, you can select the numerical variable according to' 
        'the most relevant categorical variables. Moreover, as the response variable in this case' 
        'is income, it is important to have this reference as well:',
        style={'textAlign':'center',
        'color':'green'}),
        html.P("Categorical Variable:",
        style={'color':'red'}),
        dcc.Dropdown(
        id='x-axis', 
        options=[{'value': x, 'label': x} 
                 for x in ['Attrition_Flag', 'Education_Level',  'Card_Category']],
        value=['Education_Level'], 
        multi=True
        #labelStyle={'display': 'inline-block'}
        ),
        html.P("Numerical Variable:",
        style={'color':'blue'}),
        dcc.RadioItems(
        id='y-axis', 
        options=[{'value': x, 'label': x} 
                 for x in ['Customer_Age', 'Total_Revolving_Bal', 'Total_Trans_Amt', 'Total_Trans_Ct']],
        value='Total_Trans_Ct', 
        labelStyle={'display': 'inline-block'}),
         dcc.Graph(id="box-plot")
       ]), #children #tab
     dcc.Tab(label="Categorical Variables",children=[
        dcc.Dropdown(
        id='categorical', 
        options=[{'value': x, 'label': x} 
                 for x in ['Attrition_Flag', 'Education_Level',  'Card_Category','Income_Category_final']],
        value=['Education_Level'],
         clearable=False
        #labelStyle={'display': 'inline-block'}
        ),
        dcc.Graph(id="piechart")
        ])
        ])#tabs
        
    ]
  
    elif pathname == "/page-6":
        return [
        html.P("Filter by total transactions in the account:",style={'textAlign':'center'}),
        dcc.RangeSlider(
        id='yearslider',
        min=10, max=134, step=2,
        marks={10: {'label': '10', 'style': {'color': '#77b0b1'}}, 
        20: {'label': '20'},
        50: {'label': '50'},
        80: {'label': '80'},
        100: {'label': '100'},
        120: {'label': '120'},
        134: {'label': '134','style': {'color': '#f50'}}},
        value=[20, 100]),
        html.Div(id='output-container-range-slider',style={'textAlign':'center'}),
        dcc.Graph(id="linear"),
        html.Summary(id='linear2')
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

barTab = html.Div([
    dcc.Dropdown(
        id='cat-vars',
        options=[{'label': i, 'value': i} for i in df1_cat.columns],
        value=[{'label': i, 'value': i} for i in df1_cat.columns][0],
        placeholder="Select a categorical variable: ",
        multi=True
        ),
    dcc.Graph(id='bar')
])
    
conTab = html.Div([
    dcc.Dropdown(
        id='cont-vars',
        options=[{'label': i, 'value': i} for i in df1_cont.columns],
        placeholder="Select a continuous variable: ",
        ),
    dcc.Graph(id='hist')
])

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value'))
def update_tab(selected_tab):
    if selected_tab == 'tab-cat':
        return barTab
    return conTab
     
 
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
  
  
######## pie chart

@app.callback(
    Output("piechart", "figure"), 
    [Input("categorical", "value")])

def generate_chart(x):
    fig=px.pie(df2, values='Total_Trans_Ct', names=x,
    title="Pie Chart of Categorical Variables")
    return fig    





######### linear regression
@app.callback(
    Output("linear", "figure"), 
    [Input("yearslider", "value")])
def generate_linear(trans_slider):
    tranlow, tranhigh = trans_slider
    filtertran = (df2['Total_Trans_Ct'] > tranlow) & (df2['Total_Trans_Ct'] < tranhigh)
    fig = px.scatter(df2[filtertran], x='Total_Amt_Chng_Q4_Q1', y='Total_Ct_Chng_Q4_Q1', facet_col="Attrition_Flag", color="Income_Category_final", trendline="ols")
    return fig

@app.callback(
    Output('output-container-range-slider', 'children'),
    [Input('yearslider', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)




@app.callback(
    Output("linear2", "children"), 
    [Input("yearslider", "value")])
def generate_linear(trans_slider):
    tranlow, tranhigh = trans_slider
    filtertran = (df2['Total_Trans_Ct'] > tranlow) & (df2['Total_Trans_Ct'] < tranhigh)
    fig = px.scatter(df2[filtertran], x='Total_Amt_Chng_Q4_Q1', y='Total_Ct_Chng_Q4_Q1', facet_col="Attrition_Flag", color="Income_Category_final", trendline="ols")
    results = px.get_trendline_results(fig)
    resul2=results.query("Attrition_Flag == 'Existing Customer' and Income_Category_final == 'Fistclass'").px_fit_results.iloc[0].summary()
    print(result2)
    
    









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




