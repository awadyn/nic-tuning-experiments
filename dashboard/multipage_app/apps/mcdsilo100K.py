import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State, MATCH, ALL
import plotly.express as px
import os
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from app import app

LINUX_COLS = ['i', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c1', 'c1e', 'c3', 'c6', 'c7', 'joules', 'timestamp']
EBBRT_COLS = ['i', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c3', 'c6', 'c7', 'joules', 'timestamp']
JOULE_CONVERSION = 0.00001526 #counter * constant -> JoulesOB
TIME_CONVERSION_khz = 1./(2899999*1000)

global_linux_default_df = pd.DataFrame()
global_linux_default_df_non0j = pd.DataFrame()
global_linux_default_name = []

global_linux_tuned_df = pd.DataFrame()
global_linux_tuned_df_non0j = pd.DataFrame()
global_linux_tuned_name = []

global_ebbrt_tuned_df = pd.DataFrame()
global_ebbrt_tuned_df_non0j = pd.DataFrame()
global_ebbrt_tuned_name = []

workload_loc='/scratch2/mcdsilo/mcdsilo_combined/mcdsilo_combined.csv'
log_loc='/scratch2/mcdsilo/mcdsilo_combined/'

df_comb = pd.read_csv(workload_loc, sep=' ')
df_comb = df_comb[df_comb['joules'] > 0]
df_comb = df_comb[df_comb['read_99th'] < 501]
df_comb = df_comb[df_comb['time'] > 19]
df_comb = df_comb[df_comb['target_QPS'] == 100000].copy()
axis_values = [{'label': key, 'value': key} for key in df_comb.columns]        
df_comb['time'] = df_comb['time'].astype(np.int32)

edp_fig = px.scatter(df_comb, 
                     x='time', 
                     y='joules', 
                     color='sys',
                     labels={'time': 'Time (s)', 'joules': 'Energy (Joules)'}, 
                     hover_data=['i', 'itr', 'rapl', 'dvfs', 'sys', 'num_interrupts'],
                     custom_data=['i', 'itr', 'rapl', 'dvfs', 'sys'],
                     title=f'Memcached-Silo 100K QPS')
        
layout = html.Div([
    html.H3('Memcached-Silo QPS 100K'),
    dcc.Link('Home', href='/'),
    html.Br(),
    
    dcc.Graph(
        id='mcdsilo10-edp-scatter',
        figure = edp_fig,
        style={'display': 'inline-block'},
    ),   

    html.Div(id='mcdsilo10-my-output'),
    html.P('SYS = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='mcdsilo10-sys-dd',
                 options=[
                     {'label': 'linux_tuned', 'value': 'linux_tuned'},
                     {'label': 'linux_default', 'value': 'linux_default'},
                     {'label': 'ebbrt_tuned', 'value': 'ebbrt_tuned'},
                 ],
                 value='linux_tuned',
                 style={'display': 'inline-block', 'width':'40%'}
    ),
    html.P('Run_Number = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='mcdsilo10-i-dd',                 
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
    dcc.Dropdown(id='mcdsilo10-itr-dd',
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
    dcc.Dropdown(id='mcdsilo10-dvfs-dd',
                 options=[
                     {'label': '0x1800', 'value': '0x1800'},
                     {'label': '0x1900', 'value': '0x1900'},
                     {'label': '0x1a00', 'value': '0x1a00'},
                     {'label': '0x1b00', 'value': '0x1b00'},
                     {'label': '0x1c00', 'value': '0x1c00'},
                     {'label': '0x1d00', 'value': '0x1d00'},
                     {'label': '0xffff', 'value': '0xffff'},
                 ],
                 value='0x1d00',
                 style={'display': 'inline-block', 'width':'40%'}
    ),
    html.Br(),
    html.P('RAPL = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='mcdsilo10-rapl-dd',
                 options=[
                     {'label': 135, 'value': 135},
                     {'label': 95, 'value': 95},
                     {'label': 75, 'value': 75},
                     {'label': 55, 'value': 55},
                 ],
                 value=135,
                 style={'display': 'inline-block', 'width':'40%'}
    ),
    html.P('CORE = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='mcdsilo10-core-dd',
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
                     {'label': '14', 'value': 14}
                 ],
                 value=0,
                 style={'display': 'inline-block'}
    ),
    
    html.Div([
        html.Hr(),
        html.Br()
    ]),
    html.Button(id='mcdsilo10-btn', children='Update Plots', n_clicks=0),
    html.Br(),
    dcc.Graph(
        id='mcdsilo10-timeline-1',
        style={'display': 'inline-block'},
    ),
    
    dcc.Graph(
        id='mcdsilo10-timeline-2',
        style={'display': 'inline-block'},
    ),
    
    dcc.Graph(
        id='mcdsilo10-timeline-3',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='mcdsilo10-timeline-4',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='mcdsilo10-timeline-5',
        style={'display': 'inline-block'},
    ),
    
    dcc.Graph(
        id='mcdsilo10-timeline-6',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='mcdsilo10-timeline-7',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='mcdsilo10-timeline-8',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='mcdsilo10-timeline-9',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='mcdsilo10-timeline-10',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='mcdsilo10-timeline-11',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='mcdsilo10-timeline-12',
        style={'display': 'inline-block'},
    ),
    
    html.Div([
        html.Hr(),
        html.Br()
    ]),
    
    html.Div([
        dcc.Dropdown(id='mcdsilo10-xaxis-selector-1', value='time', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='mcdsilo10-yaxis-selector-1', value='joules', style={'width':'60%'}, options=axis_values),        
        dcc.Graph(
            id='mcdsilo10-custom-scatter-1', style={'display': 'inline-block'}
        )
    ], style={'display': 'inline-block'}),
        
    html.Div([
        dcc.Dropdown(id='mcdsilo10-xaxis-selector-2', value='time', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='mcdsilo10-yaxis-selector-2', value='joules', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='mcdsilo10-custom-scatter-2', style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(id='mcdsilo10-xaxis-selector-3', value='time', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='mcdsilo10-yaxis-selector-3', value='joules', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='mcdsilo10-custom-scatter-3',
            style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),

    html.Div([
        html.Hr(),
        html.Br()
    ]),
                  
    html.Div([
        html.P('X = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='mcdsilo10-xaxis-selector-3d-1', value='time', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Y = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='mcdsilo10-yaxis-selector-3d-1', value='itr', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Z = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='mcdsilo10-zaxis-selector-3d-1', value='joules', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.Br(),
        html.P('Color = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='mcdsilo10-color-selector-3d-1', value='sys', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Symbol = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='mcdsilo10-symbol-selector-3d-1', value='dvfs', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Size = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='mcdsilo10-size-selector-3d-1', value='rapl', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.Br(),
        dcc.Graph(
            id='mcdsilo10-custom-scatter3d-1',
            style={'display': 'inline-block'},
        ),
    ]),

    html.Br(),
    html.Div([
        html.P('X = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='mcdsilo10-xaxis-selector-3d-2', value='time', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Y = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='mcdsilo10-yaxis-selector-3d-2', value='itr', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Z = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='mcdsilo10-zaxis-selector-3d-2', value='joules', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.Br(),
        html.P('Color = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='mcdsilo10-color-selector-3d-2', value='sys', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Symbol = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='mcdsilo10-symbol-selector-3d-2', value='dvfs', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Size = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='mcdsilo10-size-selector-3d-2', value='rapl', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.Br(),
        dcc.Graph(
            id='mcdsilo10-custom-scatter3d-2',
            style={'display': 'inline-block'},
        ),
    ])

                  
])

@app.callback(
    Output('mcdsilo10-i-dd', 'value'),
    Output('mcdsilo10-itr-dd', 'value'),
    Output('mcdsilo10-rapl-dd', 'value'),
    Output('mcdsilo10-dvfs-dd', 'value'),
    Output('mcdsilo10-core-dd', 'value'),
    Output('mcdsilo10-sys-dd', 'value'),
    [Input('mcdsilo10-edp-scatter', 'clickData')]
)
def update_timeline_plots1(clickData):
    print(clickData)
    custom_data = clickData['points'][0]['customdata']
    num, itr, rapl, dvfs, sys, num_interrupts= custom_data
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
            if scatter == True:
                fig1.add_trace(go.Scatter(x=global_linux_tuned_df['timestamp'], y=global_linux_tuned_df[m], name=f'{global_linux_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'red'}))
            else:
                fig1.add_trace(go.Pointcloud(x=global_linux_tuned_df['timestamp'], y=global_linux_tuned_df[m], name=f'{global_linux_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'red'}))
        if global_linux_default_df.empty == False:
            if scatter == True:
                fig1.add_trace(go.Scatter(x=global_linux_default_df['timestamp'], y=global_linux_default_df[m], name=f'{global_linux_default_name}', showlegend=True, marker={'sizemin':2, 'color':'green'}))
            else:
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
            if scatter == True:
                fig1.add_trace(go.Scatter(x=global_linux_tuned_df_non0j['timestamp'], y=global_linux_tuned_df_non0j[m], name=f'{global_linux_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'red'}))
            else:
                fig1.add_trace(go.Pointcloud(x=global_linux_tuned_df_non0j['timestamp'], y=global_linux_tuned_df_non0j[m], name=f'{global_linux_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'red'}))
        if global_linux_default_df_non0j.empty == False:
            if scatter == True:
                fig1.add_trace(go.Scatter(x=global_linux_default_df_non0j['timestamp'], y=global_linux_default_df_non0j[m], name=f'{global_linux_default_name}', showlegend=True, marker={'sizemin':2, 'color':'green'}))
            else:
                fig1.add_trace(go.Pointcloud(x=global_linux_default_df_non0j['timestamp'], y=global_linux_default_df_non0j[m], name=f'{global_linux_default_name}', showlegend=True, marker={'sizemin':2, 'color':'green'}))
        if global_ebbrt_tuned_df_non0j.empty == False:
            if scatter == True:
                fig1.add_trace(go.Scatter(x=global_ebbrt_tuned_df_non0j['timestamp'], y=global_ebbrt_tuned_df_non0j[m], name=f'{global_ebbrt_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'blue'}, mode='lines+markers'))
            else:
                fig1.add_trace(go.Pointcloud(x=global_ebbrt_tuned_df_non0j['timestamp'], y=global_ebbrt_tuned_df_non0j[m], name=f'{global_ebbrt_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'blue'}))

    return fig1

@app.callback(
#    Output('mcdsilo10-my-output', 'children'),
    Output('mcdsilo10-timeline-1', 'figure'),
    Output('mcdsilo10-timeline-2', 'figure'),
    Output('mcdsilo10-timeline-3', 'figure'),
    Output('mcdsilo10-timeline-4', 'figure'),
    Output('mcdsilo10-timeline-5', 'figure'),
    Output('mcdsilo10-timeline-6', 'figure'),
    Output('mcdsilo10-timeline-7', 'figure'),
    Output('mcdsilo10-timeline-8', 'figure'),
    Output('mcdsilo10-timeline-9', 'figure'),
    Output('mcdsilo10-timeline-10', 'figure'),
    [Input('mcdsilo10-btn', 'n_clicks')],
    [State('mcdsilo10-i-dd', 'value'),
     State('mcdsilo10-itr-dd', 'value'),
     State('mcdsilo10-rapl-dd', 'value'),
     State('mcdsilo10-dvfs-dd', 'value'),
     State('mcdsilo10-core-dd', 'value'),
     State('mcdsilo10-sys-dd', 'value')]
)
def update_plots(n_clicks, i, itr, rapl, dvfs, core, sys):
    global global_linux_default_df
    global global_linux_tuned_df
    global global_ebbrt_tuned_df
    global global_linux_default_df_non0j
    global global_linux_tuned_df_non0j
    global global_ebbrt_tuned_df_non0j    
    global global_linux_default_name
    global global_linux_tuned_name
    global global_ebbrt_tuned_name

    print(sys, i, core, itr, rapl, dvfs)
    if n_clicks == 0:
        return
    
    fname=''
    START_RDTSC=0
    END_RDTSC=0
    num_interrupts = 0
    qps = 100000
    
    if sys == 'linux_tuned' or sys == 'linux_default':        
        frdtscname = f'{log_loc}/linux.mcdsilo.rdtsc.{i}_{itr}_{dvfs}_{rapl}_{qps}'
        frdtsc = open(frdtscname, 'r')
        for line in frdtsc:
            tmp = line.strip().split(' ')
            if int(tmp[2]) > START_RDTSC:                                
                START_RDTSC = int(tmp[2])                                                                 
        frdtsc.close()

        fname = f'{log_loc}/linux.mcdsilo.dmesg.{i}_{core}_{itr}_{dvfs}_{rapl}_{qps}'
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

    return getFig('tx_bytes'), getFig('rx_bytes'), getFig('timestamp_diff'), getFig('timestamp_diff', scatter=True), getFig('joules_diff', non0j=True), getFig('joules_diff', non0j=True, scatter=True), getFig('ref_cycles_diff', non0j=True), getFig('ref_cycles_diff', non0j=True, scatter=True), getFig('instructions_diff', non0j=True), getFig('llc_miss_diff', non0j=True)

#    fig_tx_bytes = getFig('tx_bytes')
#    fig_rx_bytes = getFig('rx_bytes')
#    fig_joules_diff = getFig('joules_diff', non0j=True)
#    fig_timestamp_diff = getFig('timestamp_diff')
#    
#    return f'{n_clicks} {i} {itr} {rapl} {dvfs} {qps}', fig_tx_bytes, fig_rx_bytes, fig_joules_diff, fig_timestamp_diff

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
    #converting timestamps
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()
    df['timestamp'] = df['timestamp'] * TIME_CONVERSION_khz
    df = df[df['timestamp'] <= 20.0]
    
    # update timestamp_diff
    df['timestamp_diff'] = df['timestamp'].diff()
    df.dropna(inplace=True)
        
    ## convert global_linux_tuned_df_non0j
    df_non0j = df[df['joules'] > 0
                  & (df['instructions'] > 0)
                  & (df['cycles'] > 0)
                  & (df['ref_cycles'] > 0)
                  & (df['llc_miss'] > 0)].copy()
    df_non0j['joules'] = df_non0j['joules'] * JOULE_CONVERSION
    tmp = df_non0j[['instructions', 'ref_cycles', 'cycles', 'joules', 'timestamp', 'llc_miss', 'c1', 'c1e', 'c3', 'c6', 'c7']].diff()
    tmp.columns = [f'{c}_diff' for c in tmp.columns]
    df_non0j = pd.concat([df_non0j, tmp], axis=1)
    df_non0j.dropna(inplace=True)
    df_non0j['ref_cycles_diff'] = df_non0j['ref_cycles_diff'] * TIME_CONVERSION_khz
    #df_non0j['nonidle_timestamp_diff'] = df_non0j['ref_cycles_diff'] * TIME_CONVERSION_khz
    #global_linux_tuned_df_non0j['nonidle_frac_diff'] = global_linux_tuned_df_non0j['nonidle_timestamp_diff'] / global_linux_tuned_df_non0j['timestamp_diff']
    #print(global_linux_tuned_df_non0j['nonidle_timestamp_diff'], global_linux_tuned_df_non0j['nonidle_timestamp_diff'].shape[0])
    #print(global_linux_tuned_df_non0j['timestamp_diff'], global_linux_tuned_df_non0j['timestamp_diff'].shape[0])
    return df, df_non0j

for i in range(1, 4):
    @app.callback(
        Output('mcdsilo10-custom-scatter-'+str(i), 'figure'),
        [Input('mcdsilo10-xaxis-selector-'+str(i), 'value'),
         Input('mcdsilo10-yaxis-selector-'+str(i), 'value')]
    )
    def update_custom_scatter(xcol, ycol):
        fig = px.scatter(df_comb, 
                         x=xcol, 
                         y=ycol, 
                         color='sys',
                         hover_data=['i', 'itr', 'rapl', 'dvfs', 'sys'],
                         custom_data=['i', 'itr', 'rapl', 'dvfs', 'sys'],
                         title=f'X={xcol}\nY={ycol}')    
        return fig


for i in range(1, 2):
    @app.callback(
        Output('mcdsilo10-custom-scatter3d-'+str(i), 'figure'),
        [Input('mcdsilo10-xaxis-selector-3d-'+str(i), 'value'),
         Input('mcdsilo10-yaxis-selector-3d-'+str(i), 'value'),
         Input('mcdsilo10-zaxis-selector-3d-'+str(i), 'value'),
         Input('mcdsilo10-color-selector-3d-'+str(i), 'value'),
         Input('mcdsilo10-symbol-selector-3d-'+str(i), 'value'),
         Input('mcdsilo10-size-selector-3d-'+str(i), 'value')]
    )
    def update_custom_scatter3d(xcol, ycol, zcol, c, sym, sz):
        fig = px.scatter_3d(df_comb, 
                            x=xcol, 
                            y=ycol,
                            z=zcol,
                            color=c,
                            symbol=sym,
                            size=sz,
                            hover_data=['i', 'itr', 'rapl', 'dvfs', 'sys'],
                            custom_data=['i', 'itr', 'rapl', 'dvfs', 'sys'],
                            width=800,
                            height=800)
        return fig
