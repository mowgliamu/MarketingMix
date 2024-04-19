import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from dash import callback_context
from dash.dependencies import Input, Output, State


# Read parameters from input json file
with open('input_params.json') as f:
    data = json.load(f)['eda_univariate']

data_input = data['data_input']
date_variable = data['date_variable']
target_variable = data['target_variable']
granularity = data['granularity']


# Read and create dataframe
df = pd.read_csv(data_input)

# Datetime operations
# TODO: Whatif date is already the index?! Make a note of this.
df[date_variable] = pd.to_datetime(df[date_variable])
df.index = df[date_variable]
df.drop([date_variable], axis=1, inplace=True)
# Create Month-Year
df['month-year'] = df.index.to_period('M').strftime('%Y-%m').values


# YoY Pivot DataFrame
def get_fig3(column_name):
    '''
    '''
    if granularity == 'Weekly':
       n_row = df.shape[0]
       n_year = n_row // 52
       weeks_left = n_row % 52
       df['year'] = [None]*n_row
       df['week'] = [None]*n_row
       df['year'][-52:] = 1
       df['week'][-52:] = list(range(1, 53))
       for i in range(1, n_year + 1):
           df['year'][-52*(i+1):-52*i] = i+1
           if i == n_year:
               df['week'][:weeks_left] = list(range(1, weeks_left+1))
           else:
               df['week'][-52*(i+1):-52*i] = list(range(1, 53))
       pv_data = pd.pivot_table(df, index=df.week, columns=df.year, values=column_name, aggfunc='sum').fillna(0.)
    elif granularity == 'Monthly':
       n_row = df.shape[0]
       n_year = n_row // 12
       months_left = n_row % 12
       df['year'] = [None]*n_row
       df['month'] = [None]*n_row
       df['year'][-12:] = 1
       df['month'][-12:] = i+1
       for i in range(1, n_year + 1):
           df['year'][-12*(i+1):-12*i] = list(range(1, 13))
           if i == n_year:
               df['month'][:months_left] = list(range(1, months_left+1))
           else:
               df['month'][-12*(i+1):-12*i] = list(range(1, 13))

       pv_data = pd.pivot_table(df, index=df.month, columns=df.year, values=column_name, aggfunc='sum').fillna(0.)
    else:
        print('Choose granularity as Weekly or Monthly')

    fig = px.line(pv_data, x=pv_data.index, y=pv_data.columns)

    return fig

# Initialize App
# Play later to Stylize!
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app = dash.Dash(__name__)
app.title = 'Univariate EDA'


# Which columns to include in dropdown
# Use function / list comprehension
column_names = [{'label':i, 'value':i} for i in df.columns]


# Initialize Figures - Will be updated later
# Taking KPI_Total_Sales as default (NOTE: Must  be an argument that user can pass)
fig1 = px.line(df, x=df.index, y=target_variable, markers=True)
fig2 = px.bar(df, x='month-year', y=target_variable)
fig3 = get_fig3(target_variable)
fig4 = px.box(df, y=target_variable)
fig5 = px.violin(df, y=target_variable)
fig6 = px.histogram(df, x=target_variable)


# Create Dashborad Layout
app.layout = html.Div([
    # Div for Dropdown
    html.Div([
            dcc.Dropdown(id = 'variable-dropdown', options = column_names, value=target_variable)]),
    # Graph - 1
    html.Div([
        html.Div([
            html.H3(children='Line plot',
                    className='header_title', 
                    style={'textAlign': 'left'}),
            html.P(children='Visualization of an independent variable (such as sales or spend) at weekly granularity',
                   className='header_description',
                   style={'textAlign': 'left'}),
            dcc.Graph(id='g1', figure=fig1, style={'width': '80%'})
        ],
        style= {"display": "inline-block",
                "width": "80%",
                "margin-left": "20px",
                "verticalAlign": "top"},
        className="First Graph"),
            dcc.Textarea(id='textarea_fig1',
                         value='',
                         placeholder='Write comments here',
                         persistence=True, persistence_type='local',
                         readOnly=True,
                         style={'width': '60%', 'height': 50, 'display': 'block'}),
            html.Button('Save', id='save_button_fig1', n_clicks=0, hidden=True),
            html.Button('Edit', id='edit_button_fig1', n_clicks=0),
            html.Div(id='comments_fig1', style={'whiteSpace': 'pre-line'}),
            ]),
    # Graph - 2
    html.Div([
        html.Div([
            html.H3(children='Bar plot',
                    className='header_title', 
                    style={'textAlign': 'left'}),
            html.P(children='Month-Year aggregation of an indepedent variable, each bar is divided into weekly blocks',
                   className='header_description',
                   style={'textAlign': 'left'}),
            dcc.Graph(id='g2', figure=fig2, style={'width': '80%'})
        ],
        style= {"display": "inline-block",
                "width": "80%",
                "margin-left": "20px",
                "verticalAlign": "top"},
        className="Second Graph"),
            dcc.Textarea(id='textarea_fig2',
                         value='',
                         placeholder='Write comments here',
                         persistence=True, persistence_type='local',
                         readOnly=True,
                         style={'width': '60%', 'height': 50, 'display': 'block'}),
            html.Button('Save', id='save_button_fig2', n_clicks=0, hidden=True),
            html.Button('Edit', id='edit_button_fig2', n_clicks=0),
            html.Div(id='comments_fig2', style={'whiteSpace': 'pre-line'}),
            ]),
    # Graph - 3
    html.Div([
        html.Div([
            html.H3(children='YoY plot',
                    className='header_title', 
                    style={'textAlign': 'left'}),
            html.P(children='Year-on-Year (YoY) visualization of an independent variable at weekly granularity',
                   className='header_description',
                   style={'textAlign': 'left'}),
            dcc.Graph(id='g3', figure=fig3, style={'width': '80%'})
        ],
        style= {"display": "inline-block",
                "width": "80%",
                "margin-left": "20px",
                "verticalAlign": "top"},
        className="Third Graph"),
            dcc.Textarea(id='textarea_fig3',
                         value='',
                         placeholder='Write comments here',
                         persistence=True, persistence_type='local',
                         readOnly=True,
                         style={'width': '60%', 'height': 50, 'display': 'block'}),
            html.Button('Save', id='save_button_fig3', n_clicks=0, hidden=True),
            html.Button('Edit', id='edit_button_fig3', n_clicks=0),
            html.Div(id='comments_fig3', style={'whiteSpace': 'pre-line'}),
            ]),
    # Graph - 4
    html.Div([
        html.Div([
            html.H3(children='Box plot',
                    className='header_title', 
                    style={'textAlign': 'left'}),
            html.P(children='Box plot visualization of an independent variable',
                   className='header_description',
                   style={'textAlign': 'left'}),
            dcc.Graph(id='g4', figure=fig4, style={'width': '80%'})
        ],
        style= {"display": "inline-block",
                "width": "70%",
                "margin-left": "20px",
                "verticalAlign": "top"},
        className="Fourth Graph"),
            dcc.Textarea(id='textarea_fig4',
                         value='',
                         placeholder='Write comments here',
                         persistence=True, persistence_type='local',
                         readOnly=True,
                         style={'width': '60%', 'height': 50, 'display': 'block'}),
            html.Button('Save', id='save_button_fig4', n_clicks=0, hidden=True),
            html.Button('Edit', id='edit_button_fig4', n_clicks=0),
            html.Div(id='comments_fig4', style={'whiteSpace': 'pre-line'}),
            ]),
    # Graph - 5
    html.Div([
        html.Div([
            html.H3(children='Violin plot',
                    className='header_title', 
                    style={'textAlign': 'left'}),
            html.P(children='Violin plot visualization of an independent variable',
                   className='header_description',
                   style={'textAlign': 'left'}),
            dcc.Graph(id='g5', figure=fig5, style={'width': '80%'})
        ],
        style= {"display": "inline-block",
                "width": "70%",
                "margin-left": "20px",
                "verticalAlign": "top"},
        className="Fifth Graph"),
            dcc.Textarea(id='textarea_fig5',
                         value='',
                         placeholder='Write comments here',
                         persistence=True, persistence_type='local',
                         readOnly=True,
                         style={'width': '60%', 'height': 50, 'display': 'block'}),
            html.Button('Save', id='save_button_fig5', n_clicks=0, hidden=True),
            html.Button('Edit', id='edit_button_fig5', n_clicks=0),
            html.Div(id='comments_fig5', style={'whiteSpace': 'pre-line'}),
            ]),
    # Graph - 6
    html.Div([
        html.Div([
            html.H3(children='Histogram',
                    className='header_title', 
                    style={'textAlign': 'left'}),
            html.P(children='Histogram of an independent variable',
                   className='header_description',
                   style={'textAlign': 'left'}),
            dcc.Graph(id='g6', figure=fig6, style={'width': '80%'}),
            html.P("Number of Bins:"),
            dcc.Slider(id="number-of-bins", min=5, max=50, step=5, value=10,
                       marks={5: '5', 
                              10: '10', 
                              15: '15', 
                              20: '20', 
                              25: '25', 
                              30: '30', 
                              35: '35', 
                              40: '40', 
                              45: '45', 
                              50: "50"}),
        ],
        style= {"display": "inline-block",
                "width": "80%",
                "margin-left": "20px",
                "verticalAlign": "top"},
        className="Sixth Graph"),
            dcc.Textarea(id='textarea_fig6',
                         value='',
                         placeholder='Write comments here',
                         persistence=True, persistence_type='local',
                         readOnly=True,
                         style={'width': '60%', 'height': 50, 'display': 'block'}),
            html.Button('Save', id='save_button_fig6', n_clicks=0, hidden=True),
            html.Button('Edit', id='edit_button_fig6', n_clicks=0),
            html.Div(id='comments_fig6', style={'whiteSpace': 'pre-line'}),
            ]),
])


# Callbacks!

@app.callback(Output(component_id='g1', component_property= 'figure'),
              [Input(component_id='variable-dropdown', component_property= 'value')])
def graph_update(dropdown_value):
    fig = go.Figure([go.Scatter(x = df.index, y = df['{}'.format(dropdown_value)],\
                     line = dict(color = 'firebrick', width = 4))
                     ])
    
    fig.update_layout(yaxis_title = dropdown_value,
                      template="simple_white")
    fig.update_xaxes(rangeslider_visible=True)

    return fig


@app.callback(Output(component_id='g2', component_property= 'figure'),
              [Input(component_id='variable-dropdown', component_property= 'value')])
def graph_update(dropdown_value):
    fig = go.Figure([go.Bar(x = df['month-year'], y = df['{}'.format(dropdown_value)])])
    
    fig.update_layout(yaxis_title = dropdown_value,
                      template="simple_white")
    return fig


@app.callback(Output(component_id='g3', component_property= 'figure'),
              [Input(component_id='variable-dropdown', component_property= 'value')])
def graph_update(dropdown_value):
    fig = get_fig3(dropdown_value)
    fig.update_layout(yaxis_title = dropdown_value,
                      template="simple_white")
    fig.update_xaxes(rangeslider_visible=True)

    return fig


@app.callback(Output(component_id='g4', component_property= 'figure'),
              [Input(component_id='variable-dropdown', component_property= 'value')])
def graph_update(dropdown_value):
    fig = go.Figure([go.Box(y = df['{}'.format(dropdown_value)], name = '',\
                    boxpoints='all', fillcolor='lightseagreen', opacity=0.6)])
    
    fig.update_layout(yaxis_title = dropdown_value,
                      template="simple_white")
    fig.update_xaxes(rangeslider_visible=True)

    return fig


@app.callback(Output(component_id='g5', component_property= 'figure'),
              [Input(component_id='variable-dropdown', component_property= 'value')])
def graph_update(dropdown_value):
    fig = go.Figure([go.Violin(y = df['{}'.format(dropdown_value)], points='all',\
                     box_visible=True, line_color='black', name = '',\
                     meanline_visible=True, fillcolor='lightseagreen', opacity=0.6)])
    
    fig.update_layout(yaxis_title = dropdown_value,
                      template="simple_white")

    return fig


@app.callback(Output(component_id='g6', component_property= 'figure'),
              [Input(component_id='variable-dropdown', component_property= 'value'),
               Input(component_id='number-of-bins', component_property='value'),
              ])
def graph_update(dropdown_value, n_bins):
    fig = go.Figure([go.Histogram(x = df['{}'.format(dropdown_value)],\
                                  opacity=0.75, nbinsx=n_bins)])
    
    fig.update_layout(yaxis_title = dropdown_value,
                      template="simple_white")

    return fig


@app.callback(
    [
        Output('textarea_fig1', 'readOnly'),
        Output('save_button_fig1', 'hidden'),
        Output('edit_button_fig1', 'hidden'),
        Output('comments_fig1', 'children'),
    ],
    [
        Input('edit_button_fig1', 'n_clicks'),
        Input('save_button_fig1', 'n_clicks'),
    ],
    State('textarea_fig1', 'value'),
    prevent_initial_call=True
)
def update_output(edit, save, comment):
    trigger = callback_context.triggered_id
    if trigger == 'edit_button_fig1':
        return False, False, True, comment
    else:
        return True, True, False, comment


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

