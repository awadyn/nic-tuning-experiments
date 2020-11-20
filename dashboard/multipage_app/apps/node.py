import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State, MATCH, ALL
import plotly.express as px
import os
import plotly.graph_objects as go
import pandas as pd

#from config_local import *
#from read_agg_data import *
#from process_log_data import *

from app import app

LINUX_COLS = ['i', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c1', 'c1e', 'c3', 'c6', 'c7', 'joules', 'timestamp']
EBBRT_COLS = ['i', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c3', 'c6', 'c7', 'joules', 'timestamp']
JOULE_CONVERSION = 0.00001526 #counter * constant -> JoulesOB
TIME_CONVERSION_khz = 1./(2899999*1000)

#Home-spun caching
agg_data = {}
log_data = {}
axis_values = {}

#app = dash.Dash()

global_linux_default_df = pd.DataFrame()
global_linux_default_df_non0j = pd.DataFrame()
global_linux_default_name = []

global_linux_tuned_df = pd.DataFrame()
global_linux_tuned_df_non0j = pd.DataFrame()
global_linux_tuned_name = []

global_ebbrt_tuned_df = pd.DataFrame()
global_ebbrt_tuned_df_non0j = pd.DataFrame()
global_ebbrt_tuned_name = []

workload_loc='/scratch2/node/node_combined_11_17_2020/node_combined.csv'
log_loc='/scratch2/node/node_combined_11_17_2020/'

df_comb = pd.read_csv(workload_loc, sep=' ')
df_comb = df_comb[df_comb['joules'] > 0]
df_comb['edp'] = 0.5 * df_comb['joules'] * df_comb['time']
axis_values = [{'label': key, 'value': key} for key in df_comb.columns]        

edp_fig = px.scatter(df_comb, 
                     x='time', 
                     y='joules', 
                     color='sys',
                     labels={'time': 'Time (s)', 'joules': 'Energy (Joules)'}, 
                     hover_data=['i', 'itr', 'rapl', 'dvfs', 'sys', 'num_interrupts'],
                     custom_data=['i', 'itr', 'rapl', 'dvfs', 'sys'],
                     title=f'NodeJS EDP')

'''
td_fig = px.scatter_3d(df_comb, 
                       x='itr', 
                       y='dvfs', 
                       z='joules',
                       color='sys',
                       hover_data=['i', 'itr', 'rapl', 'dvfs', 'sys', 'num_interrupts'],
                       custom_data=['i', 'itr', 'rapl', 'dvfs', 'sys'],
                       title='X=ITR Y=DVFS Z=JOULE',
                       width=800,
                       height=800)
'''
        
layout = html.Div([
    html.H3('NodeJS'),
    dcc.Link('Home', href='/'),
    html.Br(),
    
    dcc.Graph(
        id='node-edp-scatter',
        figure = edp_fig,
        style={'display': 'inline-block'},
    ),
    
    html.Br(),
    html.P('SYS = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='node-sys-dd',                 
                 options=[
                     {'label': 'linux_tuned', 'value': 'linux_tuned'},
                     {'label': 'linux_default', 'value': 'linux_default'},
                     {'label': 'ebbrt_tuned', 'value': 'ebbrt_tuned'},
                 ],
                 value='ebbrt_tuned',
                 style={'display': 'inline-block', 'width':'40%'}
    ),    
    html.P('Run_Number = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='node-i-dd',                 
                 options=[
                     {'label': i, 'value': i} for i in range(0, 5)
                 ],
                 value=0,
                 style={'display': 'inline-block', 'width':'40%'}
    ),
    html.Br(),
    html.P('ITR = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='node-itr-dd',
                 options=[
                     {'label': i, 'value': i} for i in [
                         1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 46, 40, 50, 60, 70, 80
                     ]
                 ],
                 value=2,
                 style={'display': 'inline-block', 'width':'40%'}
    ),
    html.P('DVFS = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='node-dvfs-dd',
                 options=[
                     {'label': i, 'value': i} for i in [
                         '0xffff', '0x1d00', '0x1b00', '0x1900', '0x1700', '0x1500', '0x1300', '0x1100', '0xf00', '0xd00'
                     ]
                 ],
                 value='0x1d00',
                 style={'display': 'inline-block', 'width':'40%'}
    ),
    html.Br(),
    html.P('RAPL = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='node-rapl-dd',
                 options=[
                     {'label': i, 'value': i} for i in [
                         135, 75, 55
                     ]
                 ],
                 value=135,
                 style={'display': 'inline-block', 'width':'40%'}
    ),
    #html.Br(),
    #html.Div('Core=i rxPollCnt processCnt swEventCnt idleEventCnt'),
    html.Br(),
    html.Div(id='node-my-output'),
    html.Br(),
    html.Div(id='node-my-output2'),
    html.Br(),
    html.Div(id='node-my-output3'),
    html.Div([
        html.Hr(),
        html.Br()
    ]),
    html.Button(id='node-btn', children='Update Plots', n_clicks=0),
    html.Br(),        

    dcc.Graph(
        id='node-timeline-1',
        style={'display': 'inline-block'},
    ),
    
    dcc.Graph(
        id='node-timeline-2',
        style={'display': 'inline-block'},
    ),
    
    dcc.Graph(
        id='node-timeline-3',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='node-timeline-4',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='node-timeline-5',
        style={'display': 'inline-block'},
    ),
    
    dcc.Graph(
        id='node-timeline-6',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='node-timeline-7',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='node-timeline-8',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='node-timeline-9',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='node-timeline-10',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='node-timeline-11',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='node-timeline-12',
        style={'display': 'inline-block'},
    ),

    
    html.Div([
        html.Hr(),
        html.Br()
    ]),
    
    html.Div([
        dcc.Dropdown(id='node-xaxis-selector-1', value='time', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='node-yaxis-selector-1', value='joules', style={'width':'60%'}, options=axis_values),        
        dcc.Graph(
            id='node-custom-scatter-1', style={'display': 'inline-block'}
        )
    ], style={'display': 'inline-block'}),
        
    html.Div([
        dcc.Dropdown(id='node-xaxis-selector-2', value='time', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='node-yaxis-selector-2', value='joules', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='node-custom-scatter-2', style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(id='node-xaxis-selector-3', value='time', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='node-yaxis-selector-3', value='joules', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='node-custom-scatter-3',
            style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(id='node-xaxis-selector-4', value='time', style={'width':'60%'}, options=axis_values),
        dcc.Dropdown(id='node-yaxis-selector-4', value='joules', style={'width':'60%'}, options=axis_values),
        dcc.Graph(
            id='node-custom-scatter-4',
            style={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block'}),

    html.Div([
        html.Hr(),
        html.Br()
    ]),

    html.Div([
        html.P('X = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-xaxis-selector-3d-1', value='time', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Y = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-yaxis-selector-3d-1', value='itr', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Z = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-zaxis-selector-3d-1', value='joules', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.Br(),
        html.P('Color = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-color-selector-3d-1', value='sys', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Symbol = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-symbol-selector-3d-1', value='dvfs', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Size = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-size-selector-3d-1', value='rapl', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.Br(),
        dcc.Graph(
            id='node-custom-scatter3d-1',
            style={'display': 'inline-block'},
        ),
    ]),

    html.Br(),
    html.Div([
        html.P('X = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-xaxis-selector-3d-2', value='time', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Y = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-yaxis-selector-3d-2', value='itr', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Z = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-zaxis-selector-3d-2', value='joules', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.Br(),
        html.P('Color = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-color-selector-3d-2', value='sys', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Symbol = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-symbol-selector-3d-2', value='dvfs', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Size = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-size-selector-3d-2', value='rapl', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.Br(),
        dcc.Graph(
            id='node-custom-scatter3d-2',
            style={'display': 'inline-block'},
        ),
    ]),

    html.Br(),
    html.Div([
        html.P('X = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-xaxis-selector-3d-3', value='time', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Y = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-yaxis-selector-3d-3', value='itr', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Z = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-zaxis-selector-3d-3', value='joules', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.Br(),
        html.P('Color = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-color-selector-3d-3', value='sys', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Symbol = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-symbol-selector-3d-3', value='dvfs', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.P('Size = ', style={'display': 'inline-block'}),
        dcc.Dropdown(id='node-size-selector-3d-3', value='rapl', options=axis_values, style={'display': 'inline-block', 'width':'30%'}),
        html.Br(),
        dcc.Graph(
            id='node-custom-scatter3d-3',
            style={'display': 'inline-block'},
        ),
    ])
])

@app.callback(
    Output('node-i-dd', 'value'),
    Output('node-itr-dd', 'value'),
    Output('node-rapl-dd', 'value'),
    Output('node-dvfs-dd', 'value'),
    Output('node-sys-dd', 'value'),
    [Input('node-edp-scatter', 'clickData')]
)
def update_dropdown(clickData):
    print(clickData)
    custom_data = clickData['points'][0]['customdata']
    num, itr, rapl, dvfs, sys, num_interrupts  = custom_data
    return int(num), int(itr), int(rapl), dvfs, sys

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
#    Output(component_id='node-my-output', component_property='children'),
#    Output(component_id='node-my-output2', component_property='children'),
    Output('node-timeline-1', 'figure'),
    Output('node-timeline-2', 'figure'),
    Output('node-timeline-3', 'figure'),
    Output('node-timeline-4', 'figure'),
    Output('node-timeline-5', 'figure'),
    Output('node-timeline-6', 'figure'),
    Output('node-timeline-7', 'figure'),
    Output('node-timeline-8', 'figure'),
    Output('node-timeline-9', 'figure'),
    Output('node-timeline-10', 'figure'),
    Output('node-timeline-11', 'figure'),
    Output('node-timeline-12', 'figure'),
    [Input('node-btn', 'n_clicks')],
    [State('node-i-dd', 'value'),
     State('node-itr-dd', 'value'),
     State('node-rapl-dd', 'value'),
     State('node-dvfs-dd', 'value'),
     State('node-sys-dd', 'value')]
)
def update_timeline_plots(n_clicks, num, itr, rapl, dvfs, sys):
    global global_linux_default_df
    global global_linux_tuned_df
    global global_ebbrt_tuned_df
    global global_linux_default_df_non0j
    global global_linux_tuned_df_non0j
    global global_ebbrt_tuned_df_non0j    
    global global_linux_default_name
    global global_linux_tuned_name
    global global_ebbrt_tuned_name

    if n_clicks == 0:
        return
    
    print('n_clicks=',n_clicks)
    fname=''
    START_RDTSC=0
    END_RDTSC=0
    #output=""
    #output2="Timer() counter, fireCnt = "
    if sys == 'linux_tuned' or sys == 'linux_default':        
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

        fname = f'{log_loc}/linux.node.server.log.{num}_1_{itr}_{dvfs}_{rapl}'
        if sys == 'linux_tuned':
            global_linux_tuned_df, global_linux_tuned_df_non0j = updateDF(fname, START_RDTSC, END_RDTSC)
            global_linux_tuned_name = [sys, num, itr, rapl, dvfs]
        elif sys == 'linux_default':
            global_linux_default_df, global_linux_default_df_non0j = updateDF(fname, START_RDTSC, END_RDTSC)
            global_linux_default_name = [sys, num, itr, rapl, dvfs]
            
    elif sys == 'ebbrt_tuned':        
        frdtscname = f'{log_loc}/ebbrt_rdtsc.{num}_{itr}_{dvfs}_{rapl}'
        frdtsc = open(frdtscname, 'r')
        for line in frdtsc:
            tmp = line.strip().split(' ')
            START_RDTSC = int(tmp[0])
            END_RDTSC = int(tmp[1])
            tdiff = round(float((END_RDTSC - START_RDTSC) * TIME_CONVERSION_khz), 2)
            if tdiff > 3 and tdiff < 40:
                break
        frdtsc.close()

        '''
        fcountersname = f'{log_loc}/ebbrt_counters.{num}_{itr}_{dvfs}_{rapl}'
        fcounter = open(fcountersname, 'r')
        for line in fcounter:
            if 'core' in line:                
                output+=line.strip()+'\n'
            else:
                output2+=line
        fcounter.close()
        '''
        
        fname = f'{log_loc}/ebbrt_dmesg.{num}_1_{itr}_{dvfs}_{rapl}.csv'
        global_ebbrt_tuned_df, global_ebbrt_tuned_df_non0j = updateDF(fname, START_RDTSC, END_RDTSC, ebbrt=True)
        #update plot name
        global_ebbrt_tuned_name = [sys, num, itr, rapl, dvfs]
                
    return getFig('tx_bytes'), getFig('rx_bytes'), getFig('timestamp_diff'), getFig('timestamp_diff', scatter=True), getFig('ref_cycles_diff'), getFig('ref_cycles_diff', scatter=True), getFig('joules_diff', non0j=True),  getFig('joules_diff', non0j=True, scatter=True), getFig('instructions_diff'), getFig('instructions_diff', scatter=True), getFig('llc_miss_diff'), getFig('llc_miss_diff', scatter=True)

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

    # convert joules
    df['joules'] = df['joules'] * JOULE_CONVERSION
    
    # update diffs
    df['timestamp_diff'] = df['timestamp'].diff()
    df['instructions_diff'] = df['instructions'].diff()
    df['ref_cycles_diff'] = df['ref_cycles'].diff()
    df['ref_cycles_diff'] = df['ref_cycles_diff'] * TIME_CONVERSION_khz
    df['cycles_diff'] = df['cycles'].diff()
    df['llc_miss_diff'] = df['llc_miss'].diff()
    df.dropna(inplace=True)
    
    ## convert global_linux_tuned_df_non0j
    df_non0j = df[df['joules'] > 0
                  & (df['instructions'] > 0)
                  & (df['cycles'] > 0)
                  & (df['ref_cycles'] > 0)
                  & (df['llc_miss'] > 0)].copy()    
    tmp = df_non0j[['joules', 'c1', 'c1e', 'c3', 'c6', 'c7']].diff()
    tmp.columns = [f'{c}_diff' for c in tmp.columns]
    df_non0j = pd.concat([df_non0j, tmp], axis=1)
    df_non0j.dropna(inplace=True)
    #df_non0j['nonidle_timestamp_diff'] = df_non0j['ref_cycles_diff'] * TIME_CONVERSION_khz
    #global_linux_tuned_df_non0j['nonidle_frac_diff'] = global_linux_tuned_df_non0j['nonidle_timestamp_diff'] / global_linux_tuned_df_non0j['timestamp_diff']
    #print(global_linux_tuned_df_non0j['nonidle_timestamp_diff'], global_linux_tuned_df_non0j['nonidle_timestamp_diff'].shape[0])
    #print(global_linux_tuned_df_non0j['timestamp_diff'], global_linux_tuned_df_non0j['timestamp_diff'].shape[0])
    return df, df_non0j

for i in range(1, 5):
    @app.callback(
        Output('node-custom-scatter-'+str(i), 'figure'),
        [Input('node-xaxis-selector-'+str(i), 'value'),
         Input('node-yaxis-selector-'+str(i), 'value')]
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
        Output('node-custom-scatter3d-'+str(i), 'figure'),
        [Input('node-xaxis-selector-3d-'+str(i), 'value'),
         Input('node-yaxis-selector-3d-'+str(i), 'value'),
         Input('node-zaxis-selector-3d-'+str(i), 'value'),
         Input('node-color-selector-3d-'+str(i), 'value'),
         Input('node-symbol-selector-3d-'+str(i), 'value'),
         Input('node-size-selector-3d-'+str(i), 'value')]
    )
    def update_custom_scatter(xcol, ycol, zcol, c, sym, sz):
        print(xcol, ycol, zcol, c, sym, sz)
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

#if __name__=='__main__':
#    app.run_server(host='10.241.31.7', port='8040')
