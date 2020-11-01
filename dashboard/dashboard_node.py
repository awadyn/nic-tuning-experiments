import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State, MATCH, ALL
import plotly.express as px
import os

from config_local import *
from read_agg_data import *
from process_log_data import *

#Home-spun caching
agg_data = {}
log_data = {}
axis_values = {}

app = dash.Dash()
workload='nodejs'

df_comb, df, outlier_list = start_analysis(workload=workload)
df_comb.reset_index(inplace=True)
        
#TODO: lower dvfs for non-netpipe workloads
df_comb['Sys'] = df_comb.apply(lambda x: x['sys'] + ' default' if x['sys']=='linux' and x['dvfs'].lower()=='0xffff' and x['itr']==1 else x['sys'], axis=1)

agg_data[workload] = (df_comb, df, outlier_list)

axis_values = [{'label': key, 'value': key} for key in df_comb.columns]

#for key in df_comb.columns:
#    if 'std' not in key:
#        axis_values[key] = key
        
print(axis_values)
edp_fig = px.scatter(df_comb, 
                     x='time_mean', 
                     y='joules_mean', 
                     color='Sys',
                     labels={'time_mean': 'Time (s)', 'joules_mean': 'Energy (Joules)'}, 
                     hover_data=hover_data[workload],
                     custom_data=['itr', 'rapl', 'dvfs', 'Sys', 'sys'],
                     title=f'NodeJS EDP')


app.layout = html.Div([
    dcc.Graph(
        id='edp-scatter',
        figure = edp_fig,
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='timeline-tx-bytes',
        style={'display': 'inline-block'},
    ),
    
    dcc.Graph(
        id='timeline-rx-bytes',
        style={'display': 'inline-block'},
    ),
    
    dcc.Graph(
        id='timeline-joules_diff',
        style={'display': 'inline-block'},
    ),        
    
    dcc.Graph(
        id='timeline-timestamp_diff',
        style={'display': 'inline-block'},
    ),
    
    dcc.Graph(
        id='barplot',
        style={'display': 'inline-block'}
    ),
        
    html.Div([
        html.Hr(),
        html.Br()
    ]),
    
    html.Div([
        dcc.Dropdown(id='xaxis-selector-1', value='time_mean', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='yaxis-selector-1', value='joules_mean', style={'width':'60%'}, options=axis_values),        
        dcc.Graph(
            id='custom-scatter-1', style={'display': 'inline-block'}
        )
    ], style={'display': 'inline-block'}),
        
    html.Div([
        dcc.Dropdown(id='xaxis-selector-2', value='time_mean', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='yaxis-selector-2', value='joules_mean', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='custom-scatter-2', style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(id='xaxis-selector-3', value='time_mean', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='yaxis-selector-3', value='joules_mean', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='custom-scatter-3',
            style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(id='xaxis-selector-4', value='time_mean', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='yaxis-selector-4', value='joules_mean', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='custom-scatter-4',
            style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(id='xaxis-selector-5', value='time_mean', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='yaxis-selector-5', value='joules_mean', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='custom-scatter-5',
            style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(id='xaxis-selector-6', value='time_mean', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='yaxis-selector-6', value='joules_mean', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='custom-scatter-6',
            style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),
])

for i in range(1, 7):
    @app.callback(
        Output('custom-scatter-'+str(i), 'figure'),
        [Input('xaxis-selector-'+str(i), 'value'),
         Input('yaxis-selector-'+str(i), 'value')]
    )
    def update_custom_scatter(xcol, ycol):
        #print(xcol, ycol)
        fig = px.scatter(df_comb, 
                         x=xcol, 
                         y=ycol, 
                         color='Sys',
                         hover_data=hover_data[workload],
                         custom_data=['itr', 'rapl', 'dvfs', 'Sys', 'sys'],
                         title=f'X-AXIS={xcol}\nY-AXIS={ycol}')    
        return fig

if __name__=='__main__':
    app.run_server(debug=True)
