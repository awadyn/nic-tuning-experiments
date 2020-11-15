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

workload_loc='/scratch2/node/ebbrt/11_14_2020_19_56_45/node_combined.csv'
log_loc='/scratch2/node/ebbrt/11_14_2020_19_56_45/'

df_comb = pd.read_csv(workload_loc, sep=' ')
df_comb = df_comb[df_comb['joules'] > 0]
axis_values = [{'label': key, 'value': key} for key in df_comb.columns]        

edp_fig = px.scatter(df_comb, 
                     x='time', 
                     y='joules', 
                     color='sys',
                     labels={'time': 'Time (s)', 'joules': 'Energy (Joules)'}, 
                     hover_data=['i', 'itr', 'rapl', 'dvfs', 'sys', 'num_interrupts'],
                     custom_data=['i', 'itr', 'rapl', 'dvfs', 'sys'],
                     title=f'NodeJS EDP')

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

        
layout = html.Div([
    html.H3('NodeJS'),
    dcc.Link('Home', href='/'),
    html.Br(),
    
    dcc.Graph(
        id='edp-scatter',
        figure = edp_fig,
        style={'display': 'inline-block'},
    ),
    
    html.Br(),
    html.P('SYS = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='sys-dd',                 
                 options=[
                     {'label': 'linux_tuned', 'value': 'linux_tuned'},
                     {'label': 'linux_default', 'value': 'linux_default'},
                     {'label': 'ebbrt_tuned', 'value': 'ebbrt_tuned'},
                 ],
                 value='ebbrt_tuned',
                 style={'display': 'inline-block', 'width':'40%'}
    ),    
    html.P('Run_Number = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='i-dd',                 
                 options=[
                     {'label': i, 'value': i} for i in range(0, 5)
                 ],
                 value=0,
                 style={'display': 'inline-block', 'width':'40%'}
    ),
    html.Br(),
    html.P('ITR = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='itr-dd',
                 options=[
                     {'label': i, 'value': i} for i in [
                         2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 46, 40, 50, 60, 70, 80
                     ]
                 ],
                 value=2,
                 style={'display': 'inline-block', 'width':'40%'}
    ),
    html.P('DVFS = ', style={'display': 'inline-block'}),
    dcc.Dropdown(id='dvfs-dd',
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
    dcc.Dropdown(id='rapl-dd',
                 options=[
                     {'label': i, 'value': i} for i in [
                         135, 75, 55
                     ]
                 ],
                 value=135,
                 style={'display': 'inline-block', 'width':'40%'}
    ),
    html.Div([
        html.Hr(),
        html.Br()
    ]),
    html.Button(id='btn', children='Update Plots', n_clicks=0),
    html.Br(),
    
    html.Div(id='my-output'),

    dcc.Graph(
        id='timeline-1',
        style={'display': 'inline-block'},
    ),
    
    dcc.Graph(
        id='timeline-2',
        style={'display': 'inline-block'},
    ),
    
    dcc.Graph(
        id='timeline-3',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='timeline-4',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='timeline-5',
        style={'display': 'inline-block'},
    ),
    
    dcc.Graph(
        id='timeline-6',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='timeline-7',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='timeline-8',
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

    dcc.Graph(
        id='custom-scatter3d',
        figure = td_fig,
        style={'display': 'inline-block'},
    ),
])

@app.callback(
    Output('i-dd', 'value'),
    Output('itr-dd', 'value'),
    Output('rapl-dd', 'value'),
    Output('dvfs-dd', 'value'),
    Output('sys-dd', 'value'),
    [Input('edp-scatter', 'clickData')]
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
    ## df original
    if non0j == False:
        fig1.update_layout(xaxis_title="timestamp (s)", yaxis_title=f'{m}')
        
        if global_linux_tuned_df.empty == False:
            fig1.add_trace(go.Pointcloud(x=global_linux_tuned_df['timestamp'], y=global_linux_tuned_df[m], name=f'{global_linux_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'green'}))
        if global_linux_default_df.empty == False:
            fig1.add_trace(go.Pointcloud(x=global_linux_default_df['timestamp'], y=global_linux_default_df[m], name=f'{global_linux_default_name}', showlegend=True, marker={'sizemin':2, 'color':'red'}))
        if global_ebbrt_tuned_df.empty == False:
            if scatter == True:
                fig1.add_trace(go.Scatter(x=global_ebbrt_tuned_df['timestamp'], y=global_ebbrt_tuned_df[m], name=f'{global_ebbrt_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'blue'}))
            else:
                fig1.add_trace(go.Pointcloud(x=global_ebbrt_tuned_df['timestamp'], y=global_ebbrt_tuned_df[m], name=f'{global_ebbrt_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'blue'}))
    ## df_non0j
    else:
        fig1.update_layout(xaxis_title="timestamp (s)", yaxis_title=f'{m}')
        if global_linux_tuned_df_non0j.empty == False:
            fig1.add_trace(go.Pointcloud(x=global_linux_tuned_df_non0j['timestamp'], y=global_linux_tuned_df_non0j[m], name=f'{global_linux_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'green'}))
        if global_linux_default_df_non0j.empty == False:
            fig1.add_trace(go.Pointcloud(x=global_linux_default_df_non0j['timestamp'], y=global_linux_default_df_non0j[m], name=f'{global_linux_default_name}', showlegend=True, marker={'sizemin':2, 'color':'red'}))
        if global_ebbrt_tuned_df_non0j.empty == False:
            if scatter == True:
                fig1.add_trace(go.Scatter(x=global_ebbrt_tuned_df_non0j['timestamp'], y=global_ebbrt_tuned_df_non0j[m], name=f'{global_ebbrt_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'blue'}, mode='lines+markers'))
            else:
                fig1.add_trace(go.Pointcloud(x=global_ebbrt_tuned_df_non0j['timestamp'], y=global_ebbrt_tuned_df_non0j[m], name=f'{global_ebbrt_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'blue'}))
                
                
    return fig1
    
@app.callback(
    #Output(component_id='my-output', component_property='children'),
    Output('timeline-1', 'figure'),
    Output('timeline-2', 'figure'),
    Output('timeline-3', 'figure'),
    Output('timeline-4', 'figure'),
    Output('timeline-5', 'figure'),
    Output('timeline-6', 'figure'),
    [Input('btn', 'n_clicks')],
    [State('i-dd', 'value'),
     State('itr-dd', 'value'),
     State('rapl-dd', 'value'),
     State('dvfs-dd', 'value'),
     State('sys-dd', 'value')]
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

    print('n_clicks=',n_clicks)
    fname=''
    START_RDTSC=0
    END_RDTSC=0
    
    if sys == 'linux_tuned' or sys == 'linux_default':
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

        if sys == 'linux_tuned':
            global_linux_tuned_df = pd.read_csv(fname, sep=' ', names=LINUX_COLS)

            ## filter out rdtscs
            global_linux_tuned_df = global_linux_tuned_df[global_linux_tuned_df['timestamp'] >= START_RDTSC]
            global_linux_tuned_df = global_linux_tuned_df[global_linux_tuned_df['timestamp'] <= END_RDTSC]

            #num_interrupts = global_linux_tuned_df.shape[0]
            #converting rdtsc
            global_linux_tuned_df['timestamp'] = global_linux_tuned_df['timestamp'] - global_linux_tuned_df['timestamp'].min()
            global_linux_tuned_df['timestamp'] = global_linux_tuned_df['timestamp'] * TIME_CONVERSION_khz

            # converting joules
            global_linux_tuned_df['joules'] = global_linux_tuned_df['joules'] - global_linux_tuned_df['joules'].min()
            global_linux_tuned_df['joules'] = global_linux_tuned_df['joules'] * JOULE_CONVERSION
            #LINUX_COLS = ['i', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c1', 'c1e', 'c3', 'c6', 'c7', 'joules', 'timestamp']
            for c in ['instructions', 'ref_cycles', 'cycles', 'llc_miss', 'joules', 'timestamp', 'c1', 'c1e', 'c3', 'c6', 'c7']: 
                global_linux_tuned_df[c] = global_linux_tuned_df[c] - global_linux_tuned_df[c].min()
            

            ## convert global_linux_tuned_df_non0j
            global_linux_tuned_df_non0j = global_linux_tuned_df[global_linux_tuned_df['joules'] > 0].copy()

            ## timestamp_diff
            global_linux_tuned_df['timestamp_diff'] = global_linux_tuned_df['timestamp'].diff()
            global_linux_tuned_df.dropna(inplace=True)
            
            for c in ['instructions', 'ref_cycles', 'cycles', 'llc_miss', 'joules', 'timestamp', 'c1', 'c1e', 'c3', 'c6', 'c7']:
                global_linux_tuned_df_non0j[c] = global_linux_tuned_df_non0j[c] - global_linux_tuned_df_non0j[c].min()
                            
            tmp = global_linux_tuned_df_non0j[['instructions', 'ref_cycles', 'cycles', 'joules', 'timestamp', 'i', 'c1', 'c1e', 'c3', 'c6', 'c7']].diff()
            tmp.columns = [f'{c}_diff' for c in tmp.columns]
            global_linux_tuned_df_non0j = pd.concat([global_linux_tuned_df_non0j, tmp], axis=1)
            global_linux_tuned_df_non0j['nonidle_timestamp_diff'] = global_linux_tuned_df_non0j['ref_cycles_diff'] * TIME_CONVERSION_khz
            global_linux_tuned_df_non0j.dropna(inplace=True)
            global_linux_tuned_df_non0j['nonidle_frac_diff'] = global_linux_tuned_df_non0j['nonidle_timestamp_diff'] / global_linux_tuned_df_non0j['timestamp_diff']
            
            global_linux_tuned_name = [sys, num, itr, rapl, dvfs]
            
        elif sys == 'linux_default':
            global_linux_default_df = pd.read_csv(fname, sep=' ', names=LINUX_COLS)
            global_linux_default_df = global_linux_default_df[global_linux_default_df['timestamp'] >= START_RDTSC]
            global_linux_default_df = global_linux_default_df[global_linux_default_df['timestamp'] <= END_RDTSC]
            global_linux_default_df['timestamp'] = global_linux_default_df['timestamp'] - global_linux_default_df['timestamp'].min()
            global_linux_default_df['timestamp'] = global_linux_default_df['timestamp'] * TIME_CONVERSION_khz    
            global_linux_default_df['joules'] = global_linux_default_df['joules'] - global_linux_default_df['joules'].min()
            global_linux_default_df['joules'] = global_linux_default_df['joules'] * JOULE_CONVERSION

            #num_interrupts = global_linux_default_df.shape[0]
             
            for c in ['instructions', 'ref_cycles', 'cycles', 'llc_miss', 'joules', 'timestamp', 'c1', 'c1e', 'c3', 'c6', 'c7']: 
                global_linux_default_df[c] = global_linux_default_df[c] - global_linux_default_df[c].min()            

            ## convert global_linux_tuned_df_non0j
            global_linux_default_df_non0j = global_linux_default_df[global_linux_default_df['joules'] > 0].copy()
            global_linux_default_df['timestamp_diff'] = global_linux_default_df['timestamp'].diff()
            global_linux_default_df.dropna(inplace=True)
            for c in ['instructions', 'ref_cycles', 'cycles', 'llc_miss', 'joules', 'timestamp', 'c1', 'c1e', 'c3', 'c6', 'c7']:
                global_linux_default_df_non0j[c] = global_linux_default_df_non0j[c] - global_linux_default_df_non0j[c].min()
                            
            tmp = global_linux_default_df_non0j[['instructions', 'ref_cycles', 'cycles', 'joules', 'timestamp', 'i', 'c1', 'c1e', 'c3', 'c6', 'c7']].diff()
            tmp.columns = [f'{c}_diff' for c in tmp.columns]
            global_linux_default_df_non0j = pd.concat([global_linux_default_df_non0j, tmp], axis=1)
            global_linux_default_df_non0j['nonidle_timestamp_diff'] = global_linux_default_df_non0j['ref_cycles_diff'] * TIME_CONVERSION_khz
            global_linux_default_df_non0j.dropna(inplace=True)
            global_linux_default_df_non0j['nonidle_frac_diff'] = global_linux_default_df_non0j['nonidle_timestamp_diff'] / global_linux_default_df_non0j['timestamp_diff']            
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

        fname = f'{log_loc}/ebbrt_dmesg.{num}_1_{itr}_{dvfs}_{rapl}.csv'
        global_ebbrt_tuned_df, global_ebbrt_tuned_df_non0j = updateDF(fname, START_RDTSC, END_RDTSC, ebbrt=True)
        #update plot name
        global_ebbrt_tuned_name = [sys, num, itr, rapl, dvfs]
                
    fig1 = getFig('tx_bytes')
    fig2 = getFig('rx_bytes')    
    fig4 = getFig('timestamp_diff')
    fig10 = getFig('ref_cycles_diff')
    fig11 = getFig('instructions_diff')
    fig3 = getFig('joules_diff', non0j=True)
    #fig5 = getFig('c1_diff', non0j=True)
    #fig6 = getFig('c1e_diff', non0j=True)
    #fig7 = getFig('c3_diff', non0j=True)
    #fig8 = getFig('c7_diff', non0j=True)

    #fig12 = getFig('timestamp_diff', scatter=True)
    #fig13 = getFig('joules_diff', non0j=True, scatter=True)
    
    #return f'Selected i={num} ITR-Delay={itr} RAPL={rapl} DVFS={dvfs} SYSTEM={sys}', fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10
    #fig_test = go.Figure()
    #fig_test.update_layout(xaxis_title="timestamp_diff (s)", yaxis_title=f'ref_cycles_diff')
    #fig_test.add_trace(go.Pointcloud(x=global_ebbrt_tuned_df_non0j['timestamp_diff'], y=global_ebbrt_tuned_df_non0j['ref_cycles_diff'], name=f'{global_ebbrt_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'blue'}))
    #return f'Selected i={num} ITR-Delay={itr} RAPL={rapl} DVFS={dvfs} SYSTEM={sys} NUM_INTERRUPTS={num_interrupts}', fig1, fig2, fig3, fig4, fig10, fig11
    return fig1, fig2, fig3, fig4, fig10, fig11

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

for i in range(1, 4):
    @app.callback(
        Output('custom-scatter-'+str(i), 'figure'),
        [Input('xaxis-selector-'+str(i), 'value'),
         Input('yaxis-selector-'+str(i), 'value')]
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

#if __name__=='__main__':
#    app.run_server(host='10.241.31.7', port='8040')
