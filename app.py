import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_table as dt
import pandas as pd
import plotly.express as px
import json

df1 = pd.read_csv('https://raw.githubusercontent.com/RoberAlcaraz/First-Take-Away/main/vehicles_data.csv')





