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
df1_cat = df1[['price', 'manufacturer', 'condition', 'fuel', 'title_status', 'transmission', 'drive']]
df1_pred = df1[['year', 'odometer', 'manufacturer', 'condition', 'fuel', 'title_status', 'transmission', 'drive']]


FONTSIZE = 50
FONTCOLOR = "#F5FFFA"
BGCOLOR ="#3445DB"
models = ['Random Forest', 'KNN', 'Logistic']

min_year = min(df1['year'])
max_year = max(df1['year'])
min_odometer = min(df1['odometer'])
max_odometer = max(df1['odometer'])


## Bank Churnes Data Set

df2 = pd.read_csv('https://raw.githubusercontent.com/amaliajimenezajt/final_shiny_app/master/BankChurnersData.csv')
df2["Attrition_Flag"] = df2["Attrition_Flag"].astype("category")
df2["Gender"] = df2["Gender"].astype("category")
df2["Education_Level"] = df2["Education_Level"].astype("category")
df2["Education_Level"] = df2["Education_Level"].astype("category")
df2["Card_Category"] = df2["Card_Category"].astype("category")
df2["Income_Category_final"] = df2["Income_Category_final"].astype("category")

df2_edu = df2['Education_Level'].dropna().sort_values().unique()
opt_edu = [{'label': x , 'value': x} for x in df2_edu]

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
#### Some references
[Dash Core Components](https://dash.plot.ly/dash-core-components)  
[Dash HTML Components](https://dash.plot.ly/dash-html-components)  
[Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/l/components)  
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
                'so that it will be able to know which customers have good income in order ,to give'
                'them a better service and make them commit to stay with the bank.'),
                dcc.Markdown(markdown_text)
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
        html.P('For a start, we will do a descriptive analysis to see the behavior of our variables and how to work with them,'
        'as well as cleaning our data set. Then, we will apply different statistical tools on our data set to get the best'
        'possible information about it in order to make the best conclusions.'),
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
        html.H1('Data Description-Summary',
                style={'textAlign':'center'}),
        html.Br(),
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
       # html.Div(id='my-div', style={'display': 'none'}),
        
          # dcc.Dropdown(
          #           id='my-multi-dropdown',
          #           options=opt_edu,
          #           value=df2_edu[0],
          #           multi=True
          #       ),
        html.P("Filter by total transactions in the account:",style={'textAlign':'center'}),
        dcc.RangeSlider(
        id='yearslider',
        step=2,
        min=min(df2['Total_Trans_Ct']),
        max=max(df2['Total_Trans_Ct']), 
        marks={10: {'label': '10', 'style': {'color': '#77b0b1'}}, 
        20: {'label': '20'},
        50: {'label': '50'},
        80: {'label': '80'},
        100: {'label': '100'},
        120: {'label': '120'},
        134: {'label': '134','style': {'color': '#f50'}}},
        value=[20,40]),
        html.Div(id='output-container-range-slider',style={'textAlign':'center'}),
        dcc.Graph(id="linear"),
        # html.Button('Update filter', id='my-button')]),
         dt.DataTable(
                id='my-table',
                columns=[{"name": i, "id": i} for i in df2.columns]
            )
            
            
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
        value=['manufacturer'],
        placeholder="Select a categorical variable: ",
        multi=True
        ),
    dcc.Graph(id='bar')
])
    

conTab = html.Div([
    dcc.Dropdown(
        id='cont-vars',
        options=[{'label': i, 'value': i} for i in df1_cont.columns],
        value=['year'],
        placeholder="Select a continuous variable: ",
    ),
    html.Div(id='cont-opt'),
    dcc.Graph(id='hist1'),
    dcc.Graph(id='hist2'),
])

@app.callback(
    Output('cont-opt', 'children'),
    Input('cont-vars', 'value'))
def set_var_options(selected_var):
  if selected_var == 'odometer':
    return html.Div([
    dcc.RangeSlider(
      id='range_cont2',
      min=min_odometer,
      max=max_odometer,
      step=10000,
      value=[min_odometer, max_odometer],
    )
    ])
    
  return html.Div([
    dcc.RangeSlider(
      id='range_cont1',
      min=min_year,
      max=max_year,
      step=1000,
      value=[min_year, max_year],
    )
    ])


@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value'))
def update_tab(selected_tab):
    if selected_tab == 'tab-cat':
        return barTab
    return conTab
     
 
# @app.callback(
#     Output('hist', 'figure'),
#     [Input('cont-vars', 'value'),
#     Input('range-cont', 'value')])
# def update_hist(selected_var, range_value):
#   df = df1[df1[selected_var].between(range_value[0], range_value[1])]
#     fig = px.histogram(df, x=selected_var)
#     return fig

@app.callback(
  Output('hist1', 'figure'),
  Input('range-cont1', 'value'))
def update_hist(range_value):
  df = df1[df1['year'].between(range_value[0], range_value[1])]
  fig = px.histogram(df, x='year')
  return fig
  
@app.callback(
  Output('hist2', 'figure'),
  Input('range-cont2', 'value'))
def update_hist(range_value):
  df = df1[df1['odometer'].between(range_value[0], range_value[1])]
  fig = px.histogram(df, x='odometer')
  return fig
     

@app.callback(
    Output('bar', 'figure'),
    Input('cat-vars', 'value'))
def update_bar(selected_var):
    fig = px.bar(df1, 
            x=selected_var, 
            color_discrete_map={
              "Easy": "magenta",
              "Diff": "goldenrod"
            })
    return fig


    
@app.callback(
    [
        Output("precision", 'value'),
        Output("recall", 'value'),
        Output("accuracy", 'value'),
    ],
    [
        Input("predictors", "value"),
        Input("slider", "value"),
        Input("select_models", "value")        
    ]
)
def buildModel(pred, slider, bestModel):
    
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

@app.callback(
    Output('prueba', 'children'),
    Input('drop', 'value'))
def update_output(value):
    return 'You have selected "{}"'.format(value)

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


# @app.callback(
#     Output('my-div', 'children'),
#     [Input('my-button', 'n_clicks')],
#     [State('yearslider', 'value')])
#     
# def update_data2(n_clicks, slider_range):
#     if (slider_range and len(slider_range) == 2):
#         l, h = slider_range
#     else :
#         l, h = min(df2['Total_Trans_Ct']), max(df2['Total_Trans_Ct']);
#     df = df2[df2['Total_Trans_Ct'].between(l,h)].to_json(orient='split', date_format='iso')
#     return json.dumps(df)
#   
#########################

# @app.callback(
#     [Output('my-graph', 'figure')],
#     [Input('my-div', 'children'),
#      Input('my-multi-dropdown', 'value')]
# )
# 
# 
# def update_output_graph2(data, input_value):
#     if data is None:
#         return {}, {}
#     dataset = json.loads(data)
#     df = pd.read_json(dataset, orient='split')
#     return  {
#                 'data': [
#                     go.Scatter(
#                         x=df[df['Education_Level'] == i]['Total_Trans_Amt'] if i in input_value else [],
#                         y=df[df['Education_Level'] == i]['Total_Trans_Ct'] if i in input_value else [],
#                         text=df[df['Education_Level'] == i]['customer'],
#                         mode='markers',
#                         opacity=0.7,
#                         marker={
#                             'size': 15,
#                             'line': {'width': 0.5, 'color': 'white'}
#                         },
#                         customer=i
#                     ) for i in df2_edu
#                 ],
#                 'layout': go.Layout(
#                     xaxis={ 'title': 'Total_Trans_Amt'},
#                     yaxis={'title': 'Total_Trans_Ct'},
#                     hovermode='closest',
#                     dragmode='lasso'
#                 )
#             }


#########################


# @app.callback(
#     [Output('yearslider', 'min'),
#      Output('yearslider', 'max'),
#      Output('yearslider', 'value'),
#      Output('yearslider', 'marks')],
#     [Input('my-multi-dropdown', 'value')]
# )
# def update_slider2(input_value):
#     def round(x):
#         return int(x) if x % 2 < 2 else x
# 
#     s = pd.Series(input_value, name='Education_Level')
#     data = df2[df2.Education_Level.isin(s)]['Total_Trans_Ct']
# 
#     min = round(data.min())
#     max = round(data.max())
#     mean = round(data.mean())
#     low = round((min + mean)/2)
#     high = round((max + mean) / 2)
#     marks = {min: {'label': str(min), 'style': {'color': '#77b0b1'}},
#              max: {'label': str(max), 'style': {'color': '#77b0b1'}}}
#     return min, max,  [low, high], marks

###############################

# @app.callback(
#     Output('my-table', 'data'),
#     [Input('linear', 'selectedData')])
# def display_selected_data(selected_data):
#     if selected_data is None or len(selected_data) == 0:
#         return []
# 
#     points = selected_data['points']
#     if len(points) == 0:
#         return []
# 
#     names = [x['text'] for x in points]
#     return df2[df2['customer'].isin(names)].to_dict("rows")
# 



######### linear regression
@app.callback(
    Output("linear", "figure"),
    [Input("yearslider", "value")])
def generate_linear(trans_slider):
    tranlow, tranhigh = trans_slider
    filtertran = (df2['Total_Trans_Ct'] > tranlow) & (df2['Total_Trans_Ct'] < tranhigh)
    fig = px.scatter(df2[filtertran], x='Total_Amt_Chng_Q4_Q1', y='Total_Ct_Chng_Q4_Q1', facet_col="Attrition_Flag", color="Income_Category_final", trendline="ols")
    return fig


#######################

@app.callback(
    Output('output-container-range-slider', 'children'),
    [Input('yearslider', 'value')])
def update_output(value):
    return 'You have selected "{}"'.print(value)
  
##########

# @app.callback(
#     Output('my-table', 'data'),
#     [Input('linear', 'selectedData')])
# def display_selected_data(selected_data):
#     if selected_data is None or len(selected_data) == 0:
#         return []
# 
#     points = selected_data['points']
#     if len(points) == 0:
#         return []
# 
#     names = [x['text'] for x in points]
#     return df2[df2['customer'].isin(names)].to_dict("rows")
# 





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




