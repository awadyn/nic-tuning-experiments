import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State, MATCH, ALL
import plotly.express as px
import os
import plotly.graph_objects as go
import pandas as pd

from app import app

#from config_local import *
#from read_agg_data import *
#from process_log_data import *

LINUX_COLS = ['i', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c1', 'c1e', 'c3', 'c6', 'c7', 'joules', 'timestamp']
EBBRT_COLS = ['i', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c3', 'c6', 'c7', 'joules', 'timestamp']
JOULE_CONVERSION = 0.00001526 #counter * constant -> JoulesOB
TIME_CONVERSION_khz = 1./(2899999*1000)

#Home-spun caching
agg_data = {}
log_data = {}
axis_values = {}

global_linux_default_df = pd.DataFrame()
global_linux_default_df_non0j = pd.DataFrame()
global_linux_default_name = []

global_linux_tuned_df = pd.DataFrame()
global_linux_tuned_df_non0j = pd.DataFrame()
global_linux_tuned_name = []

global_ebbrt_tuned_df = pd.DataFrame()
global_ebbrt_tuned_df_non0j = pd.DataFrame()
global_ebbrt_tuned_name = []

workload='mcd'
workload_loc='/scratch2/han/asplos_2021_datasets/mcd/mcd_combined_11_9_2020/mcd_combined.csv'
log_loc='/scratch2/han/asplos_2021_datasets/mcd/mcd_combined_11_9_2020/'

df_comb = pd.read_csv(workload_loc, sep=' ')
df_comb = df_comb[df_comb['joules'] > 0]
df_comb = df_comb[df_comb['read_99th'] < 501]
df_comb_200k = df_comb[df_comb['target_QPS'] == 200000].copy()
axis_values = [{'label': key, 'value': key} for key in df_comb.columns]        

edp_fig_200k = px.scatter(df_comb_200k, 
                     x='time', 
                     y='joules', 
                     color='sys',
                     labels={'time': 'Time (s)', 'joules': 'Energy (Joules)'}, 
                     hover_data=['i', 'itr', 'rapl', 'dvfs', 'sys'],
                     custom_data=['i', 'itr', 'rapl', 'dvfs', 'sys'],
                     title=f'Memcached 200K QPS')

#td_fig = px.scatter_3d(df_comb, 
#                       x='itr', 
#                       y='dvfs', 
#                       z='joules',
#                       color='sys',
#                       hover_data=['i', 'itr', 'rapl', 'dvfs', 'sys', 'instructions', 'c1', 'c7'],
#                       custom_data=['i', 'itr', 'rapl', 'dvfs', 'sys', 'instructions', 'c1', 'c7'],
#                       title='X=ITR Y=DVFS Z=JOULE',
#                       width=800,
#                       height=800)

        
layout = html.Div([
    html.H3('Memcached QPS 200K'),
    dcc.Link('Home', href='/'),
    html.Br(),
    
    dcc.Graph(
        id='edp-scatter-200k',
        figure = edp_fig_200k,
        style={'display': 'inline-block'},
    ),   

    html.Div(id='my-output'),
    html.P('SYS = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='msys-dd',                 
                 options=[
                     {'label': 'linux_tuned', 'value': 'linux_tuned'},
                     {'label': 'linux_default', 'value': 'linux_default'},
                     {'label': 'ebbrt_tuned', 'value': 'ebbrt_tuned'},
                 ],
                 value='linux_tuned',
                 style={'display': 'inline-block', 'width':'40%'}
    ),    
    html.P('Run_Number = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='mi-dd',                 
                 options=[
                     {'label': 0, 'value': 0},
                     {'label': 1, 'value': 1},
                     {'label': 2, 'value': 2},
                     {'label': 3, 'value': 3},
                     {'label': 4, 'value': 4},
                     {'label': 5, 'value': 5},
                 ],
                 value=0,
                 style={'display': 'inline-block', 'width':'40%'}
    ),
    html.Br(),
    html.P('ITR = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='mitr-dd',
                 options=[
                     {'label': 50, 'value': 50},
                     {'label': 100, 'value': 100},
                     {'label': 200, 'value': 200},
                     {'label': 300, 'value': 300},
                     {'label': 400, 'value': 400},
                     {'label': 1, 'value': 1},
                 ],
                 value=50,
                 style={'display': 'inline-block', 'width':'40%'}
    ),
    html.P('DVFS = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='mdvfs-dd',
                 options=[
                     {'label': '0xd00', 'value': '0xd00'},
                     {'label': '0xf00', 'value': '0xf00'},
                     {'label': '0x1100', 'value': '0x1100'},
                     {'label': '0x1300', 'value': '0x1300'},
                     {'label': '0x1500', 'value': '0x1500'},
                     {'label': '0x1900', 'value': '0x1900'},
                     {'label': '0x1b00', 'value': '0x1b00'},
                     {'label': '0x1d00', 'value': '0x1d00'},
                     {'label': '0xffff', 'value': '0xffff'},
                 ],
                 value='0x1d00',
                 style={'display': 'inline-block', 'width':'40%'}
    ),
    html.Br(),
    html.P('RAPL = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='mrapl-dd',
                 options=[
                     {'label': 135, 'value': 135},
                     {'label': 95, 'value': 95},
                     {'label': 55, 'value': 55},
                 ],
                 value=135,
                 style={'display': 'inline-block', 'width':'40%'}
    ),

    html.P('QPS = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='mqps-dd',
                 options=[
                     {'label': 200000, 'value': 200000},
                     {'label': 400000, 'value': 400000},
                     {'label': 600000, 'value': 600000},
                 ],
                 value=200000,
                 style={'display': 'inline-block', 'width':'40%'}
    ),
    html.Br(),
    html.P('CORE = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='mcore-dd',
                 options=[
                     {'label': '0', 'value': 0},
                     {'label': '1', 'value': 1},
                     {'label': '2', 'value': 2},
                     {'label': '3', 'value': 3},
                     {'label': '4', 'value': 4},
                     {'label': '5', 'value': 5},
                     {'label': '6', 'value': 6},
                     {'label': '7', 'value': 7},
                     {'label': '8', 'value': 8},
                     {'label': '9', 'value': 9},
                     {'label': '10', 'value': 10},
                     {'label': '11', 'value': 11},
                     {'label': '12', 'value': 12},
                     {'label': '13', 'value': 13},
                     {'label': '14', 'value': 14},
                     {'label': '15', 'value': 15},
                     {'label': 'all', 'value': 66}
                 ],
                 value=0,
                 style={'display': 'inline-block'}
    ),
    
    html.Div([
        html.Hr(),
        html.Br()
    ]),
    html.Button(id='mbtn', children='Update Plots', n_clicks=0),
    html.Br(),
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
        id='timeline-joules_diff_pointcloud',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='timeline-timestamp_diff',
        style={'display': 'inline-block'},
    ),
    
    dcc.Graph(
        id='timeline-timestamp_diff_pointcloud',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='timeline-nonidle_diff',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='timeline-test1',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='timeline-test2',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='timeline-instructions_diff',
        style={'display': 'inline-block'},
    ),
    
    html.Div([
        html.Hr(),
        html.Br()
    ]),
    
    html.Div([
        dcc.Dropdown(id='xaxis-selector-1', value='time', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='yaxis-selector-1', value='joules', style={'width':'60%'}, options=axis_values),        
        dcc.Graph(
            id='custom-scatter-1', style={'display': 'inline-block'}
        )
    ], style={'display': 'inline-block'}),
        
    html.Div([
        dcc.Dropdown(id='xaxis-selector-2', value='time', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='yaxis-selector-2', value='joules', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='custom-scatter-2', style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(id='xaxis-selector-3', value='time', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='yaxis-selector-3', value='joules', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='custom-scatter-3',
            style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),

    html.Div([
        html.Hr(),
        html.Br()
    ]),

#    dcc.Graph(
#        id='custom-scatter3d',
#        figure = td_fig,
#        style={'display': 'inline-block'},
#    ),
])

@app.callback(
    Output('mi-dd', 'value'),
    Output('mitr-dd', 'value'),
    Output('mrapl-dd', 'value'),
    Output('mdvfs-dd', 'value'),
    Output('mcore-dd', 'value'),
    Output('msys-dd', 'value'),
    [Input('edp-scatter-200k', 'clickData')]
)
def update_timeline_plots1(clickData):
    print(clickData)
    custom_data = clickData['points'][0]['customdata']
    num, itr, rapl, dvfs, sys= custom_data
    return int(num), int(itr), int(rapl), dvfs, 0, sys

def getFig(m, non0j=False, scatter=False):
    global global_linux_default_df
    global global_linux_tuned_df
    global global_ebbrt_tuned_df
    global global_linux_default_df_non0j
    global global_linux_tuned_df_non0j
    global global_ebbrt_tuned_df_non0j
    global global_linux_default_name
    global global_linux_tuned_name
    global global_ebbrt_tuned_name

    fig1 = go.Figure()
    if non0j == False:
        fig1.update_layout(xaxis_title="timestamp (s)", yaxis_title=f'{m}')
        
        if global_linux_tuned_df.empty == False:
            fig1.add_trace(go.Pointcloud(x=global_linux_tuned_df['timestamp'], y=global_linux_tuned_df[m], name=f'{global_linux_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'red'}))
        if global_linux_default_df.empty == False:
            fig1.add_trace(go.Pointcloud(x=global_linux_default_df['timestamp'], y=global_linux_default_df[m], name=f'{global_linux_default_name}', showlegend=True, marker={'sizemin':2, 'color':'green'}))
        if global_ebbrt_tuned_df.empty == False:
            if scatter == True:
                fig1.add_trace(go.Scatter(x=global_ebbrt_tuned_df['timestamp'], y=global_ebbrt_tuned_df[m], name=f'{global_ebbrt_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'blue'}))
            else:
                fig1.add_trace(go.Pointcloud(x=global_ebbrt_tuned_df['timestamp'], y=global_ebbrt_tuned_df[m], name=f'{global_ebbrt_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'blue'}))
    ## df_non0j
    else:
        fig1.update_layout(xaxis_title="timestamp (s)", yaxis_title=f'{m}')
        if global_linux_tuned_df_non0j.empty == False:
            fig1.add_trace(go.Pointcloud(x=global_linux_tuned_df_non0j['timestamp'], y=global_linux_tuned_df_non0j[m], name=f'{global_linux_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'red'}))
        if global_linux_default_df_non0j.empty == False:
            fig1.add_trace(go.Pointcloud(x=global_linux_default_df_non0j['timestamp'], y=global_linux_default_df_non0j[m], name=f'{global_linux_default_name}', showlegend=True, marker={'sizemin':2, 'color':'green'}))
        if global_ebbrt_tuned_df_non0j.empty == False:
            if scatter == True:
                fig1.add_trace(go.Scatter(x=global_ebbrt_tuned_df_non0j['timestamp'], y=global_ebbrt_tuned_df_non0j[m], name=f'{global_ebbrt_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'blue'}, mode='lines+markers'))
            else:
                fig1.add_trace(go.Pointcloud(x=global_ebbrt_tuned_df_non0j['timestamp'], y=global_ebbrt_tuned_df_non0j[m], name=f'{global_ebbrt_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'blue'}))

    return fig1

@app.callback(
    Output('my-output', 'children'),
    Output('timeline-tx-bytes', 'figure'),
    Output('timeline-rx-bytes', 'figure'),
    Output('timeline-joules_diff', 'figure'),
    Output('timeline-timestamp_diff', 'figure'),
    [Input('mbtn', 'n_clicks')],
    [State('mi-dd', 'value'),
     State('mitr-dd', 'value'),
     State('mrapl-dd', 'value'),
     State('mdvfs-dd', 'value'),
     State('mqps-dd', 'value'),
     State('mcore-dd', 'value'),
     State('msys-dd', 'value')]
)
def update_plots(n_clicks, i, itr, rapl, dvfs, qps, core, sys):
    global global_linux_default_df
    global global_linux_tuned_df
    global global_ebbrt_tuned_df
    global global_linux_default_df_non0j
    global global_linux_tuned_df_non0j
    global global_ebbrt_tuned_df_non0j    
    global global_linux_default_name
    global global_linux_tuned_name
    global global_ebbrt_tuned_name

    print(sys, i, core, itr, rapl, dvfs, qps)
    fname=''
    START_RDTSC=0
    END_RDTSC=0
    num_interrupts = 0
    
    if sys == 'linux_tuned' or sys == 'linux_default':        
        frdtscname = f'{log_loc}/linux.mcd.rdtsc.{i}_{itr}_{dvfs}_{rapl}_{qps}'
        frdtsc = open(frdtscname, 'r')
        for line in frdtsc:
            tmp = line.strip().split(' ')
            if int(tmp[2]) > START_RDTSC:                                
                START_RDTSC = int(tmp[2])
            
            if END_RDTSC == 0:                                
                END_RDTSC = int(tmp[3])
            elif END_RDTSC < int(tmp[3]):
                END_RDTSC = int(tmp[3])                                                            
        frdtsc.close()

        fname = f'{log_loc}/linux.mcd.dmesg.{i}_{core}_{itr}_{dvfs}_{rapl}_{qps}'
        if sys == 'linux_tuned':
            global_linux_tuned_df, global_linux_tuned_df_non0j = updateDF(fname, START_RDTSC, END_RDTSC)
            #update plot name
            global_linux_tuned_name = [sys, core, i, itr, rapl, dvfs, qps]
        elif sys == 'linux_default':
            global_linux_default_df, global_linux_default_df_non0j = updateDF(fname, START_RDTSC, END_RDTSC)
            #update plot name
            global_linux_default_name = [sys, core, i, itr, rapl, dvfs, qps]
        else:
            print("unknown sys", sys)
    elif sys == 'ebbrt_tuned':
        frdtscname = f'{log_loc}/ebbrt_rdtsc.{i}_{itr}_{dvfs}_{rapl}_{qps}'
        frdtsc = open(frdtscname, 'r')
        for line in frdtsc:
            tmp = line.strip().split(' ')
            START_RDTSC = int(tmp[0])
            END_RDTSC = int(tmp[1])
            break    
        frdtsc.close()

        fname = f'{log_loc}/ebbrt_dmesg.{i}_{core}_{itr}_{dvfs}_{rapl}_{qps}.csv'        
        global_ebbrt_tuned_df, global_ebbrt_tuned_df_non0j = updateDF(fname, START_RDTSC, END_RDTSC, ebbrt=True)
        #update plot name
        global_ebbrt_tuned_name = [sys, core, i, itr, rapl, dvfs, qps]
            
    fig_tx_bytes = getFig('tx_bytes')
    fig_rx_bytes = getFig('rx_bytes')
    fig_joules_diff = getFig('joules_diff', non0j=True)
    fig_timestamp_diff = getFig('timestamp_diff')
    
    return f'{n_clicks} {i} {itr} {rapl} {dvfs} {qps}', fig_tx_bytes, fig_rx_bytes, fig_joules_diff, fig_timestamp_diff

def updateDF(fname, START_RDTSC, END_RDTSC, ebbrt=False):
    df = pd.DataFrame()
    if ebbrt:
        df = pd.read_csv(fname, sep=' ', names=EBBRT_COLS, skiprows=1)
        df['c1'] = 0
        df['c1e'] = 0
    else:
        df = pd.read_csv(fname, sep=' ', names=LINUX_COLS)
    ## filter out timestamps
    df = df[df['timestamp'] >= START_RDTSC]
    df = df[df['timestamp'] <= END_RDTSC]
    #converting timestamps
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()
    df['timestamp'] = df['timestamp'] * TIME_CONVERSION_khz
    # update timestamp_diff
    df['timestamp_diff'] = df['timestamp'].diff()
    df.dropna(inplace=True)

    # update total num_interrupts
    num_interrupts = df.shape[0]
        
    ## convert global_linux_tuned_df_non0j
    df_non0j = df[df['joules'] > 0
                  & (df['instructions'] > 0)
                  & (df['cycles'] > 0)
                  & (df['ref_cycles'] > 0)
                  & (df['llc_miss'] > 0)].copy()
    df_non0j['joules'] = df_non0j['joules'] * JOULE_CONVERSION
    tmp = df_non0j[['instructions', 'ref_cycles', 'cycles', 'joules', 'timestamp', 'c1', 'c1e', 'c3', 'c6', 'c7']].diff()
    tmp.columns = [f'{c}_diff' for c in tmp.columns]
    df_non0j = pd.concat([df_non0j, tmp], axis=1)
    df_non0j.dropna(inplace=True)
    df_non0j['nonidle_timestamp_diff'] = df_non0j['ref_cycles_diff'] * TIME_CONVERSION_khz            
    #global_linux_tuned_df_non0j['nonidle_frac_diff'] = global_linux_tuned_df_non0j['nonidle_timestamp_diff'] / global_linux_tuned_df_non0j['timestamp_diff']
    #print(global_linux_tuned_df_non0j['nonidle_timestamp_diff'], global_linux_tuned_df_non0j['nonidle_timestamp_diff'].shape[0])
    #print(global_linux_tuned_df_non0j['timestamp_diff'], global_linux_tuned_df_non0j['timestamp_diff'].shape[0])
    return df, df_non0j
