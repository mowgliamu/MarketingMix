import sys
import json
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

with open('input_params.json') as f:
    data = json.load(f)


# Read data from json
data_input = data['eda_bivariate']['data_input']
date_variable = data['eda_bivariate']['date_variable']
splom_list = data['eda_bivariate']['splom_list']


# Read and create dataframe
df = pd.read_csv(data_input)

# Datetime operations : module
df[date_variable] = pd.to_datetime(df[date_variable])
df.index = df[date_variable]
df['month-year'] = df.index.to_period('M').strftime('%Y-%m').values

# Initialize APP
app = dash.Dash(__name__)
app.title = 'Multivariate EDA'


def get_splom(splom_list):

    sp_dimensions = []
    for var in splom_list:
        sp_dimensions.append(dict(label=var, values=df[var]))

    splom = go.Figure(data=go.Splom(
                    dimensions = sp_dimensions, marker=dict(line_color='white', line_width=0.5)))

    return splom


splom = get_splom(splom_list)

splom.update_layout(
    title='Corrleation between Media Spends and Total Sales',
    dragmode='select',
    width=1200,
    height=1200,
    hovermode='closest',
    template='simple_white',
)


# Basic layout for one graph
app.layout = html.Div([
                       # Graph - 2
                       html.H1('Multivariate EDA'),
                       html.H3('Analyze relationship between two or more variables'),
                       html.Button(" + Add Figure", id="add-graph", n_clicks=0),
                       html.Div(),
                       html.Div(id='block_graph', children=[]),
                       # Graph - 2
                       html.H3('Scatter Plot Matrix (SPLOM) - Correlation'),
                       html.Div(),
                       dcc.Graph(id='splom',figure=splom)
                     ])


@app.callback( Output('block_graph', 'children'),
               [Input('add-graph', 'n_clicks')],
               [State('block_graph', 'children')])

def add_xy_graph(n_clicks, children):
    
    new_zone_graph = html.Div(
        style={'width': '80%', 
               'display': 'inline-block', 
               'outline': 'thin lightgrey solid', 
               'padding': 10},
        children=[
                  # First dropdown - X selection
                  dcc.Dropdown(
                               id={
                                   'type':'Selection_variable_X',
                                   'index': n_clicks
                                   },
                               options=[{'label':i, 'value':i} for i in df.columns],
                               value = None
                              ),
                  # Second dropdown  - Y selection
                  dcc.Dropdown(
                               id={
                                   'type':'Selection_variable_Y',
                                   'index': n_clicks
                                   },
                               options=[{'label':i, 'value':i} for i in df.columns],
                               multi=True,
                               value = None,
                               clearable=False,
                              ), 
                  # Third dropdown  - Chart type
                  dcc.Dropdown(
                               id={
                                   'type':'Chart_type',
                                   'index': n_clicks
                                   },
                               options=[
                                    {"value": "line", "label": "Line chart"},
                                    {"value": "scatter", "label": "Scatter chart"},
                                    {"value": "area", "label": "Area chart"},
                               ],
                               value = "line"
                              ), 
                  # Fourth dropdown  - Plot sum of y-axis lines!
                  dcc.Dropdown(
                               id={
                                   'type':'Plot_sum',
                                   'index': n_clicks
                                   },
                               options=[
                                    {"value": True, "label": "Plot Sum"},
                                    {"value": False, "label": "No Sum Plot"},
                               ],
                               value = False
                              ), 
                  # Fifth dropdown  - Single axes or multiple axes
                  dcc.Dropdown(
                               id={
                                   'type':'Secondary_axes',
                                   'index': n_clicks
                                   },
                               options=[
                                    {"value": "multiple", "label": "Add secondary axes"},
                                    {"value": "single", "label": "Single axes"},
                               ],
                               value = "single"
                              ), 
                  # Graph
                  dcc.Graph(
                            id ={'type': 'Graphic',
                                 'index': n_clicks}
                            ),
                 ])
    
    # Append figure to children    
    children.append(new_zone_graph)

    return children
 
@app.callback( Output({'type':'Graphic', 'index':MATCH},'figure'),
               [Input({'type':'Selection_variable_X', 'index':MATCH}, 'value'),
                Input({'type':'Selection_variable_Y', 'index':MATCH}, 'value'),
                Input({'type':'Chart_type', 'index':MATCH}, 'value'),
                Input({'type':'Plot_sum', 'index':MATCH}, 'value'),
                Input({'type':'Secondary_axes', 'index':MATCH}, 'value')]
            )
def generate_chart(x_axis, y_axis, graph, plot_sum, add_secondary_axis):

    # Initialize
    fig = go.Figure()

    if not x_axis:
        raise PreventUpdate
    if not y_axis:
        raise PreventUpdate
    if not graph:
        raise PreventUpdate
    if graph == "line":
        n_lines = len(y_axis)
        # Add individual lines
        for i in range(n_lines):
            fig.add_trace(go.Scatter(x=df[x_axis], y=df[y_axis[i]], mode='lines', name=y_axis[i]))
        # Plot sum if true!
        if plot_sum:
            fig.add_trace(go.Scatter(x=df[x_axis], y=df[y_axis].sum(axis=1).values, mode='lines', line={'dash':'dashdot', 'color':'black'},  name='Sum'))
        else:
            pass
        # Update Layout
        fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1),
                template='simple_white',
                )
        fig.update_xaxes(rangeslider_visible=True)

        # Add Secondary axes
        if n_lines == 2 and add_secondary_axis == 'multiple':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[x_axis], y=df[y_axis[0]], mode='lines', name=y_axis[0]))
            fig.add_trace(go.Scatter(x=df[x_axis], y=df[y_axis[1]], mode='lines', name=y_axis[1], yaxis="y2"))
            fig.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1),
                    yaxis=dict(
                        title=y_axis[0],
                    ),
                    yaxis2=dict(
                    title=y_axis[1],
                    anchor="x",
                    overlaying="y",
                    side="right"
                    ),
                    template='simple_white')
            fig.update_xaxes(rangeslider_visible=True)
    elif graph == "area":
        n_lines = len(y_axis)
        # Add individual lines
        for i in range(n_lines):
            fig.add_trace(go.Scatter(x=df[x_axis], y=df[y_axis[i]], fill='tozeroy', mode='lines', name=y_axis[i]))
        # Plot sum if true!
        if plot_sum:
            fig.add_trace(go.Scatter(x=df[x_axis], y=df[y_axis].sum(axis=1).values, mode='lines', line={'dash':'dashdot', 'color':'black'},  name='Sum'))
        else:
            pass
        # Update Layout
        fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1),
                template='simple_white',
                )
        fig.update_xaxes(rangeslider_visible=True)
    elif graph == "scatter":
        n_lines = len(y_axis)
        if n_lines == 1:
            #fig = make_subplots(rows=1,cols=1)
            fig.add_trace(go.Scatter(x=df[x_axis], y=df[y_axis[0]], mode='markers', name=''))
            X, Y = df[x_axis].copy(), df[y_axis[0]].copy()
            correlation = np.corrcoef(X, Y)
            slope, y_int=np.polyfit(X, Y, 1)
            LR="Linear Fit: {:,.3e}x + {:,.3e}".format(slope, y_int)
            rmse=np.sqrt(sum(slope*X+y_int-Y)**2)
            best_fit = slope*X+y_int
            fig.add_trace(go.Scatter(
                name='Best Line Fit',
                x=X,
                y=best_fit,
                mode='lines',
                line_color='green',
                line_width=2,
                textposition='top right'))
            fig.add_annotation(
                x=2,
                y=5,
                xref="x",
                yref="y",
                text='Correlation'+" {:.2f}".format(correlation[0][1]),
                showarrow=False,
                font=dict(
                    family="Courier New, monospace",
                    size=16,
                    color="#ffffff"
                    ),
                align="center",
                ax=20,
                ay=-30,
                bordercolor="#c7c7c7",
                borderwidth=2,
                borderpad=4,
                bgcolor="#ff7f0e",
                opacity=0.8
                )
                #legendgroup='group1',
                #legendgrouptitle_text='Trendline'))
            fig.update_layout(xaxis={'title':x_axis}, yaxis={'title':y_axis[0]})
        else:
            for i in range(n_lines):
                fig.add_trace(go.Scatter(x=df[x_axis], y=df[y_axis[i]], mode='markers', name=y_axis[i]))
        fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1),
                template='simple_white',
                )
    else:
        pass

    return fig
  


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
