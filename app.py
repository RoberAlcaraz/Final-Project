##########################################################
# Amalia Jiménez Toledano and Roberto Jesús Alcaraz Molina
##########################################################

import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_table as dt
# py_install("dash-daq")
import dash_daq as daq
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import statsmodels.api as sm
import json

# py_install("scikit-learn")
import lightgbm as lgb
import numpy as np
from sklearn import preprocessing
import plotly.figure_factory as ff
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_score,accuracy_score


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
df1_cat = df1[['manufacturer', 'condition', 'fuel', 'title_status', 'transmission', 'drive']]
df1_pred = df1[['year', 'odometer', 'manufacturer', 'condition', 'fuel', 'title_status', 'transmission', 'drive']]


FONTSIZE = 50
FONTCOLOR = "#F5FFFA"
BGCOLOR ="#3445DB"
models = ['Random Forest', 'KNN', 'Logistic']

min_year = min(df1['year'])
max_year = df1['year'].max()
min_odometer = min(df1['odometer'])
max_odometer = df1['odometer'].max()


## Bank Churnes Data Set

df2 = pd.read_csv('https://raw.githubusercontent.com/amaliajimenezajt/final_shiny_app/master/BankChurnersData.csv')
df2["Attrition_Flag"] = df2["Attrition_Flag"].astype("category")
df2["Gender"] = df2["Gender"].astype("category")
df2["Education_Level"] = df2["Education_Level"].astype("category")
df2["Education_Level"] = df2["Education_Level"].astype("category")
df2["Card_Category"] = df2["Card_Category"].astype("category")
df2["Income_Category_final"] = df2["Income_Category_final"].astype("category")

df2_card = df2['Card_Category'].dropna().sort_values().unique()
opt_card = [{'label': x , 'value': x} for x in df2_card]

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr(
            [html.Td(row[col]) for col in row.index.values]
        ) for index, row in dataframe.head(max_rows).iterrows()]
    )
    
#### REFENRENCES

markdown_text = '''
# Some references
[Dash Core Components](https://dash.plot.ly/dash-core-components)  

[Dash HTML Components](https://dash.plot.ly/dash-html-components)  

[Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/l/components)  

[Dash DAQ](https://dash.plotly.com/dash-daq)

[Dash Example Regression](https://github.com/plotly/dash-regression)

[Nav Bar tutorial](https://morioh.com/p/68e6c284a59c)
'''



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

s_header={
  'backgroundColor': 'rgb(30, 30, 30)'
  }
s_cell={
  'backgroundColor': 'rgb(50, 50, 50)',
  'color': 'white'
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
                dbc.NavLink("Bank Churners: Regression Plot-Summary", href="/page-6", active="exact"),
                dbc.NavLink("References", href="/page-7", active="exact"),
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
                html.Br(),
                dcc.Tabs(id = "tabs-i", value = "tab-v", children=[
                  dcc.Tab(label = "Vehicles data set", value="tab-v"),
                  dcc.Tab(label = "Bank Churners data set", value="tab-b")
                  ]
                ),
                html.Div(id = "tabs-intro"),
        ]

        return html.Div(
        [
          html.H1('Introduction', style={'textAlign':'center'}),
          html.Br(),
          html.P('Craigslist is the world’s largest collection of used vehicles for sale.'
          'This data set includes every used vehicle entry within the United States on'
          'Craiglist, from the year 1900 until today. This data set has been taken'
          'from the website'),
          html.Br(),
          html.P('A bank manager is interested in predicting the annual income of his or her clients account holder.'
          'For the new year, the bank has decided to create a new service depending on this income,'
          'so that it will be able to know which customers have good income in order to give'
          'them a better service and make them commit to stay with the bank.')
        ])


    elif pathname == "/page-1":
        return [
        html.H1('Data description: Craiglist vehicles', style={'textAlign':'center'}),
        html.Br(),
        html.P('For a start, we will do a descriptive analysis to see the behavior of our variables and how to work with them, '
        'as well as cleaning our data set. Then, we will apply different statistical tools on our data set to get the best '
        'possible information about it in order to make the best conclusions.'),
        html.P(
          "In the following table, you will be able to see 10 rows of the complete "
          "data set. Moreover, you can sort the data in every column, eliminate some "
          "of them and search for some value in each column."
        ),
        html.Br(),
        dt.DataTable(
            id='datatable-interactivity1',
            columns=[
                {"name": i, "id": i, "deletable": True, "selectable": True} for i in df1.columns
            ],
            data=df1.to_dict('records'),
            filter_action="native",
            sort_action="native",
            page_size= 10,
            style_header=s_header,
            style_cell=s_cell,
        ),
        html.Div(id='datatable-interactivity-container1')
                ]
    elif pathname == "/page-2":
        return [
        html.H1('Descriptive analysis', style={'textAlign':'center'}),
        html.Br(),
        html.P(
          "In this panel, we observe some plots for the continuous and categorical variables: "
        ),
        dcc.Tabs(id = "tabs", value = "tab-cont", children=[
            dcc.Tab(label = "Continuous variables", value="tab-cont"),
            dcc.Tab(label = "Categorical variables", value="tab-cat")
        ]),
        html.Div(id = "tabs-content")
        ]
        
        
    elif pathname == "/page-3":
        return [
        html.H1('Statistical models', style={'textAlign':'center'}),
        html.Br(),
        html.P(
          "In this page we can apply some statistical models to classify the price. "
          "Firstly, we will select the proportion that is employed to train the model, "
          "then we will choose all the predictors that will be in the model and finally "
          "we can select between Random Forest, KNN or Logistic Regression. Below, "
          "we can see the results (precision, recall and accuracy) for each method."
        ),
        html.Hr(),
        html.Br(),
        html.Div([
          daq.Slider(
            id = 'slider',
            min=50,
            max=90,
            value=70,
            handleLabel={"showCurrentValue": True,"label": "SPLIT"},
            step=10
            ),
        ], className="row flex-display", style={'padding-left':'35%'}
        ),
        html.Br(),
        dcc.Dropdown(
            id="predictors",
            options = [{'label':x, 'value':x} for x in df1_pred],
            multi=True,
            placeholder="Select the variables that will be in the model: ",
            clearable=False,
            className="dcc_control",
            ),
        html.Br(),
        dcc.Dropdown(
            id="select_models",
            placeholder="Select the model: ",
            options = [{'label':x, 'value':x} for x in models],
            clearable=False,
            className="dcc_control",
            ),
        html.Br(),
        html.P(
          "When you are ready, just press the button!"
        ),
        html.Button(id='submit-button-state', n_clicks=0, children='Go!'),
        html.Br(),
        html.Div([
          html.Div([
            
            daq.LEDDisplay(
              id='precision',
              label="PRECISION",
              value=0.00,
              size=FONTSIZE,
              color = FONTCOLOR,
              backgroundColor=BGCOLOR
              ),
              
            daq.LEDDisplay(
              id='recall',
              value=0.00,
              label = "RECALL",
              size=FONTSIZE,
              color = FONTCOLOR,
              backgroundColor=BGCOLOR
              ),
              
            daq.LEDDisplay(
              id='accuracy',
              value=0.00,
              label = "ACCURACY",
              size=FONTSIZE,
              color = FONTCOLOR,
              backgroundColor=BGCOLOR
              ),
              
            ], 
            className="row flex-display", style={'textAlign':'center','margin': 'auto','width': '50%','border':'3px solid green','padding': '10px'}
            ),
            ]
            ),
            
        # html.Table([
        # html.Tr([html.Td(['Precision: ']), html.Td(id='precision')]),
        # html.Tr([html.Td(['Recall: ']), html.Td(id='recall')]),
        # html.Tr([html.Td(['Accuracy: ']), html.Td(id='accuracy')]),
        # ]),
        ]
        
    elif pathname == "/page-4":
        return [
        html.H1('Data Description: Bank Churners',
                style={'textAlign':'center'}),
        html.Br(),
        html.P('For this data set, I have decided to make the graphs of both numerical and categorical variables,'
        'so you can make the boxplot of the numerical variables and the pie chat of the categorical variables.'
        'Finally, you will be able to perform a regression analysis of the total number of customer transactions'
        'for both Ct and Amt.'),
        html.Br(),
        html.Div(
            [dt.DataTable(
                id='datatable-interactivity',
                columns=[
                    {"name": i, "id": i, "deletable": True, "selectable": True} for i in df2.columns
                    ],
                    data=df2.to_dict('records'),
                    page_size= 10,
                    filter_action="native",
                    sort_action="native",
                    style_header=s_header,
                    style_cell=s_cell,
                    )
        ])
        ]

    elif pathname == "/page-5":
        return [
        html.H1('Plot numerical vs categorical',
        style={'textAlign':'center'}),
        html.Br(),
        dcc.Tabs([
          dcc.Tab(label="Numerical Variables",children=[
          html.P('In the following graph, you can select the numerical variable according to' 
        'the most relevant categorical variables. Moreover, as the response variable in this case' 
        'is income, it is important to have this reference as well:'
        ),
        html.P("Categorical Variable:"),
        dcc.Dropdown(
        id='x-axis', 
        options=[{'value': x, 'label': x} 
                 for x in ['Attrition_Flag', 'Education_Level',  'Card_Category']],
        value=['Education_Level'], 
        multi=True
        #labelStyle={'display': 'inline-block'}
        ),
        html.P("Numerical Variable:"),
        dcc.RadioItems(
        id='y-axis', 
        options=[{'value': x, 'label': x} 
                 for x in ['Customer_Age', 'Total_Revolving_Bal', 'Total_Trans_Amt', 'Total_Trans_Ct']],
        value='Total_Trans_Ct', 
        labelStyle={'display': 'inline-block'}),
         dcc.Graph(id="box-plot")
       ]), #children #tab
     dcc.Tab(label="Categorical Variables",children=[
       html.P('In the following graph, you can select the categorical variable'
       'to display the proportion of the sample in this class:'),
        dcc.Dropdown(
        id='categorical', 
        options=[{'value': x, 'label': x} 
                 for x in ['Attrition_Flag', 'Education_Level',  'Card_Category','Income_Category_final']],
        value=['Education_Level']
        #labelStyle={'display': 'inline-block'}
        ),
        dcc.Graph(id="piechart")
        ])
        ])#tabs
        
    ]
  
    elif pathname == "/page-6":
        return [
        html.P('It is very important to take into account the number of transactions'
        'in the account in relation to the amount, so in the following graph,'
        'you can see the customers and label them by card type'
        ),
        html.Div(id='my-div', style={'display': 'none'}),
        dcc.Dropdown(
          id='my-multi-dropdown',
          options=opt_card,
          value=df2_card[0],
          multi=True
        ),
        html.P("Filter by total transactions in the account:",style={'textAlign':'center'}),
        dcc.RangeSlider(
          id='my-slider',
          step=0.1,
          min=min(df2['Total_Trans_Ct']),
          max=df2['Total_Trans_Amt'].max(),
        ),
        html.P('Click the button to upload the Transiction Acount:'),
        html.Button('Update filter', id='my-button'),
        dcc.Graph(id="my-graph"),
        
        html.P('Click on the graph and check the customer information:'),
        dt.DataTable(
          id='my-table',
          columns=[{"name": i, "id": i} for i in df2.columns],
          style_header=s_header,
          style_cell=s_cell,
        ),
      ]
        
    elif pathname == "/page-7":
        return [
        html.Div([
          dcc.Markdown(markdown_text)
        ], style={'textAlign':'center'})
        
        ]       
    
    
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )
    
    
# CALLBACKS
##################################################################
@app.callback(
    Output('tabs-intro', 'children'),
    Input('tabs-i', 'value'))
def update_tab(selected_tab):
    if selected_tab == 'tab-b':
        return bTab
    return vTab
  
vTab = html.Div([
  html.Br(),
  html.P('Craigslist is the world’s largest collection of used vehicles for sale.'
  'This data set includes every used vehicle entry within the United States on'
  ' Craiglist, from the year 1900 until today. This data set has been taken'
  ' from the website Kaggle. A summary of the variables can be seen below: '),
  html.Ul("- price: indicates the price of the vehicle."),
  html.Ul("- year: indicates the year of the vehicle."),
  html.Ul("- manufacturer: indicates the class of the vehicle."),
  html.Ul("- condition: condition of the car (like new, good, etc)."),
  html.Ul("- fuel: fuel that consumes each car."),
  html.Ul("- odometer: indicates the kms of a car."),
  html.Ul("- title_status: indicates if the car is able to drive or not."),
  html.Ul("- transmission: indicates the transmission of the cars."),
  html.Ul("- drive: indicates the wheel drive of the vehicle."),
])

bTab = html.Div([
  html.Br(),
  html.P('A bank manager is interested in predicting the annual income of his or her clients account holder.'
  ' For the new year, the bank has decided to create a new service depending on this income, '
  'so that it will be able to know which customers have good income in order to give '
  'them a better service and make them commit to stay with the bank.'),
  
  html.Ul("- Attrition_Flag: if the account is closed then 1 else 0."),
  html.Ul("- Customer_Age: Customer's Age in Years. "),
  html.Ul("- Gender: M=Male, F=Female. "),
  html.Ul("- Education_Level: Educational Qualification of the account holder (example: high school, college graduate, etc.). "),
  html.Ul("- Card-Category: Type of Card (Blue, Silver, Gold, Platinum)."),
  html.Ul("- Credit_Limit: Credit Limit on the Credit Card. "),
  html.Ul("- Avg_Open_To_Buy: Open to Buy Credit Line (Average of last 12 months). "),
  html.Ul("- Total_Trans_Amt: Total Transaction Amount (Last 12 months). "),
  html.Ul("- Total_Trans_Ct: Total Transaction Count (Last 12 months). "),
  html.Ul("- Total_Ct_Chng_Q4_Q1: Change in Transaction Count (Q4 over Q1). "),
  html.Ul("- Income_Category:Demographic variable - Annual Income Category of the account holder (< 40K, 40K - 60K, 60K - 0K, 80K-120K, > "),
])

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
  html.Br(),
  html.P(
    "In this tab, we have a barplot for the selected categorical variable, dividing each "
    "category taking into account the price. Also, we can select more than one variables."
  ),
  dcc.Dropdown(
      id='cat-vars',
      options=[{'label': i, 'value': i} for i in df1_cat.columns],
      value=['manufacturer'],
      placeholder="Select a categorical variable: ",
      multi=True
      ),
  dcc.Graph(id='bar')
])
    

conTab = html.Div([
  html.Br(),
  html.P(
    "In the following input, we can select the continuous variable for which we want "
    "to see their histogram."
  ),
  dcc.Dropdown(
      id='cont-vars',
      options=[{'label': i, 'value': i} for i in df1_cont.columns],
      placeholder="Select a continuous variable: ",
  ),
  html.Br(),
  html.Div(id='cont-opt', children=[]),
  dcc.Graph(id='hist'),
])

@app.callback(
    Output('cont-opt', 'children'),
    Input('cont-vars', 'value'))
def set_var_options(selected_var):
  if selected_var == "odometer":
    return html.Div([
    html.P(
    "Since we have some outliers in our variables, we can select a smaller interval"
    "to avoid them:"
    ),
    dcc.RangeSlider(
      id='range_cont',
      min=min_odometer,
      max=max_odometer,
      step=10000,
      value=[min_odometer, max_odometer],
      allowCross=False,
    ),
    ])
  elif selected_var == "year":
    return html.Div([
    html.P(
    "Since we have some outliers in our variables, we can select a smaller interval"
    "to avoid them:"
    ),
    dcc.RangeSlider(
      id='range_cont',
      min=min_year,
      max=max_year,
      step=10,
      value=[min_year, max_year],
      allowCross=False,
    )
    ])
  return None
    
@app.callback(
    Output('hist', 'figure'),
    [Input('cont-vars', 'value'),
    Input('range_cont', 'value')])
def update_hist(selected_var, range_value):
  df = df1[df1[selected_var].between(range_value[0], range_value[1])]
  fig = px.histogram(df, x=selected_var)
  return fig



@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value'))
def update_tab(selected_tab):
    if selected_tab == 'tab-cat':
        return barTab
    return conTab


@app.callback(
    Output('bar', 'figure'),
    Input('cat-vars', 'value'))
def update_bar(selected_var):
    fig = px.bar(df1, 
            x=selected_var, 
            color="price")
    return fig


@app.callback(
    [
      Output("precision", 'value'),
      Output("recall", 'value'),
      Output("accuracy", 'value'),
    ],
    [
      Input('submit-button-state', 'n_clicks'),
      State("predictors", "value"),
      State("slider", "value"),
      State("select_models", "value")        
    ]
)
def buildModel(n_clikcs, pred, slider, bestModel):
    
    target = df1['price']
    independent = df1[pred]
    cat = list(independent.select_dtypes(include=['category']).columns)
    
    le = preprocessing.LabelEncoder()
    for i in cat:
        independent[i] = le.fit_transform(independent[i])
    

    X = pd.DataFrame(independent)
    y = pd.DataFrame(target)
    

    trainX, testX, trainy, testy = train_test_split(X, y, train_size= slider/100, random_state=2)

    if bestModel == 'Logistic':
        mod = LogisticRegression()
    elif bestModel == 'KNN':
        mod = KNeighborsClassifier()
    elif bestModel == 'Random Forest':
        mod = RandomForestClassifier()
    
    mod.fit(trainX, trainy.values.ravel())
        
    
    lr_probs = mod.predict_proba(testX)
    yhat = mod.predict(testX)
    
    lr_probs = lr_probs[:, 1]
    
    # precision tp / (tp + fp)
    precision = round(precision_score(testy, yhat,pos_label='Easy'),2)
    # recall: tp / (tp + fn)
    recall = round(recall_score(testy, yhat,pos_label='Easy'),2)
    accuracy = round(accuracy_score(testy, yhat)*100,1)
    
    return str(precision),str(recall),str(accuracy)


#################################################### table df2

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


##################


@app.callback(
    Output('my-div', 'children'),
    [Input('my-button', 'n_clicks')],
    [State('my-slider', 'value')])
def update_data(n_clicks, slider_range):
    if (slider_range and len(slider_range) == 2):
        l, h = slider_range
    else :
        l, h = min(df2['Total_Trans_Ct']), df2['Total_Trans_Ct'].max();
    df = df2[df2['Total_Trans_Ct'].between(l,h)].to_json(orient='split', date_format='iso')
    return json.dumps(df)

#########################

@app.callback(
    Output('my-graph', 'figure'),
    [Input('my-div', 'children'),
     Input('my-multi-dropdown', 'value')]
)
def update_output_graph(data, input_value):
    if data is None:
        return {}, {}
    dataset = json.loads(data)
    df = pd.read_json(dataset, orient='split')
    return  {
                'data': [
                    go.Scatter(
                        x=df[df['Card_Category'] == i]['Total_Trans_Amt'] if i in input_value else [],
                        y=df[df['Card_Category'] == i]['Total_Trans_Ct'] if i in input_value else [],
                        text=df[df['Card_Category'] == i]['customer'],
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=i
                    ) for i in df2_card
                ],
                'layout': go.Layout(
                    xaxis={'title': 'Total_Trans_Amt'},
                    yaxis={'title': 'Total_Trans_Ct'},
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                    legend={'x': 0, 'y': 1},
                    hovermode='closest',
                    dragmode='lasso'
                )
            }

#########################


@app.callback(
    [Output('my-slider', 'min'),
     Output('my-slider', 'max'),
     Output('my-slider', 'value'),
     Output('my-slider', 'marks')],
    [Input('my-multi-dropdown', 'value')]
)
def update_slider(input_value):
    def round(x):
        return int(x) if x % 2 < 2 else x

    s = pd.Series(input_value, name='Card_Category')
    data = df2[df2.Card_Category.isin(s)]['Total_Trans_Ct']

    min = round(data.min())
    max = round(data.max())
    mean = round(data.mean())
    low = round((min + mean)/2)
    high = round((max + mean) / 2)
    marks = {min: {'label': str(min), 'style': {'color': '#77b0b1'}},
             max: {'label': str(max), 'style': {'color': '#77b0b1'}}}
    return min, max,  [low, high], marks

###############################
# 
@app.callback(
    Output('my-table', 'data'),
    [Input('my-graph', 'selectedData')])
def display_selected_data(selected_data):
    if selected_data is None or len(selected_data) == 0:
        return []

    points = selected_data['points']
    if len(points) == 0:
        return []

    names = [x['text'] for x in points]
    return df2[df2['customer'].isin(names)].to_dict("rows")



######## linear regression


# @app.callback(
#     Output("my-graph", "figure"),
#     [Input("yearslider", "value")])
# def generate_linear(trans_slider):
#     tranlow, tranhigh = trans_slider
#     filtertran = (df2['Total_Trans_Ct'] > tranlow) & (df2['Total_Trans_Ct'] < tranhigh)
#     fig = px.scatter(df2[filtertran], x='Total_Amt_Chng_Q4_Q1', y='Total_Ct_Chng_Q4_Q1', facet_col="Attrition_Flag", color="Income_Category_final", trendline="ols")
#     return fig


#######################

# @app.callback(
#     Output('output-container-range-slider', 'children'),
#     [Input('yearslider', 'value')])
# def update_output(value):
#     return 'You have selected "{}"'.print(value)
# 






if __name__=='__main__':
    app.run_server()


