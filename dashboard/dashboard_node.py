import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State, MATCH, ALL
import plotly.express as px
import os
#import pandas as pd

from config_local import *
from read_agg_data import *
from process_log_data import *

LINUX_COLS = ['i', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c1', 'c1e', 'c3', 'c6', 'c7', 'joules', 'timestamp']

#Home-spun caching
agg_data = {}
log_data = {}
axis_values = {}

app = dash.Dash()
workload='nodejs'
workdload_loc='/home/han/github/nic-tuning-experiments/dashboard/summary_data/node_combined_11_1_2020.csv'
log_loc='/home/han/github/nic-tuning-experiments/dashboard/log_data/node_combined_11_1_2020'

df_comb_all = pd.read_csv(workdload_loc, sep=' ')
axis_values = [{'label': key, 'value': key} for key in df_comb_all.columns]        
#print(df_comb.shape[0], axis_values)
df_comb = df_comb_all.copy()
df_comb = df_comb[df_comb['i'] == 1]

edp_fig = px.scatter(df_comb, 
                     x='time', 
                     y='joule', 
                     color='sys',
                     labels={'time': 'Time (s)', 'joules': 'Energy (Joules)'}, 
                     hover_data=['i', 'itr', 'rapl', 'dvfs', 'sys'],
                     custom_data=['i', 'itr', 'rapl', 'dvfs', 'sys'],
                     title=f'NodeJS EDP')


'''
df_comb, df, outlier_list = start_analysis(workload=workload)
df_comb.reset_index(inplace=True)        
#TODO: lower dvfs for non-netpipe workloads
df_comb['Sys'] = df_comb.apply(lambda x: x['sys'] + ' default' if x['sys']=='linux' and x['dvfs'].lower()=='0xffff' and x['itr']==1 else x['sys'], axis=1)
agg_data[workload] = (df_comb, df, outlier_list)
axis_values = [{'label': key, 'value': key} for key in df_comb.columns]        
print(axis_values)
edp_fig = px.scatter(df_comb, 
                     x='time_mean', 
                     y='joule', 
                     color='Sys',
                     labels={'time_mean': 'Time (s)', 'joule': 'Energy (Joules)'}, 
                     hover_data=hover_data[workload],
                     custom_data=['itr', 'rapl', 'dvfs', 'Sys', 'sys'],
                     title=f'NodeJS EDP')
'''

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

    html.Div(id='my-output'),
    html.Div([
        html.Hr(),
        html.Br()
    ]),
    
    html.Div([
        dcc.Dropdown(id='xaxis-selector-1', value='time', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='yaxis-selector-1', value='joule', style={'width':'60%'}, options=axis_values),        
        dcc.Graph(
            id='custom-scatter-1', style={'display': 'inline-block'}
        )
    ], style={'display': 'inline-block'}),
        
    html.Div([
        dcc.Dropdown(id='xaxis-selector-2', value='time', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='yaxis-selector-2', value='joule', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='custom-scatter-2', style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(id='xaxis-selector-3', value='time', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='yaxis-selector-3', value='joule', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='custom-scatter-3',
            style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(id='xaxis-selector-4', value='time', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='yaxis-selector-4', value='joule', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='custom-scatter-4',
            style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(id='xaxis-selector-5', value='time', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='yaxis-selector-5', value='joule', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='custom-scatter-5',
            style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(id='xaxis-selector-6', value='time', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='yaxis-selector-6', value='joule', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='custom-scatter-6',
            style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),
])

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Output('timeline-tx-bytes', 'figure'),
    [Input('edp-scatter', 'clickData')]
)
def update_timeline_lots(clickData):
    custom_data = clickData['points'][0]['customdata']
    num, itr, rapl, dvfs, sys = custom_data

    fname=''
    START_RDTSC=0
    END_RDTSC=0
    
    if sys == 'linux_tuned':
        fname = f'{log_loc}/linux.node.server.log.{num}_1_{itr}_{dvfs}_{rapl}'
        frdtscname = f'{log_loc}/linux.node.server.rdtsc.{num}_1_{itr}_{dvfs}_{rapl}'
        frdtsc = open(frdtscname, 'r')
        for line in frdtsc:
            tmp = line.strip().split(' ')
            START_RDTSC = int(tmp[1])
            END_RDTSC = int(tmp[2])
            tdiff = round(float((END_RDTSC - START_RDTSC) * TIME_CONVERSION_khz), 2)
            if tdiff > 3 and tdiff < 40:
                break
        frdtsc.close()
        
        df = pd.read_csv(fname, sep=' ', names=LINUX_COLS)
        df = df[df['timestamp'] >= START_RDTSC]
        df = df[df['timestamp'] <= END_RDTSC]
        df['timestamp'] = df['timestamp'] - df['timestamp'].min()
        df['timestamp'] = df['timestamp'] * TIME_CONVERSION_khz
        
        df['joules'] = df['joules'] - df['joules'].min()
        df['joules'] = df['joules'] * JOULE_CONVERSION

        m = 'tx_bytes'
        fig1 = px.scatter(df, x='timestamp', y=m, labels={'timestamp': 'Time(s)', m: m}, title=f'Timeline for metric = {m}')
        
        return f'Selected i={num} ITR-Delay={itr} RAPL={rapl} DVFS={dvfs} SYSTEM={sys}', fig1
    else:
        return None, None


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
                         color='sys',
                         hover_data=['i', 'itr', 'rapl', 'dvfs', 'sys'],
                         custom_data=['i', 'itr', 'rapl', 'dvfs', 'sys'],
                         title=f'X={xcol}\nY={ycol}')    
        return fig

if __name__=='__main__':
    app.run_server(debug=True)
