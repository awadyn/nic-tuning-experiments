import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State, MATCH, ALL
import plotly.express as px
import os
import plotly.graph_objects as go

from config_local import *
from read_agg_data import *
from process_log_data import *

LINUX_COLS = ['i', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c1', 'c1e', 'c3', 'c6', 'c7', 'joules', 'timestamp']
EBBRT_COLS = ['i', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c3', 'c6', 'c7', 'joules', 'timestamp']

#Home-spun caching
agg_data = {}
log_data = {}
axis_values = {}

app = dash.Dash()

global_linux_default_df = pd.DataFrame()
global_linux_default_df_non0j = pd.DataFrame()
global_linux_default_name = []

global_linux_tuned_df = pd.DataFrame()
global_linux_tuned_df_non0j = pd.DataFrame()
global_linux_tuned_name = []

global_ebbrt_tuned_df = pd.DataFrame()
global_ebbrt_tuned_df_non0j = pd.DataFrame()
global_ebbrt_tuned_name = []

workload='nodejs'
workload_loc='/scratch2/han/asplos_2021_datasets/node/node_combined_11_4_2020/node_combined_11_10_2020.csv'
log_loc='/scratch2/han/asplos_2021_datasets/node/node_combined_11_4_2020'

df_comb = pd.read_csv(workload_loc, sep=' ')
df_comb = df_comb[df_comb['joule'] > 0]
axis_values = [{'label': key, 'value': key} for key in df_comb.columns]        
#print(df_comb.shape[0], axis_values)
#df_comb['inter_interrupt_diff_mean'] = (df_comb['timestamp'].diff()).mean()

edp_fig = px.scatter(df_comb, 
                     x='time', 
                     y='joule', 
                     color='sys',
                     labels={'time': 'Time (s)', 'joules': 'Energy (Joules)'}, 
                     hover_data=['i', 'itr', 'rapl', 'dvfs', 'sys', 'instructions', 'c1', 'c7'],
                     custom_data=['i', 'itr', 'rapl', 'dvfs', 'sys', 'instructions', 'c1', 'c7'],
                     title=f'NodeJS EDP')

td_fig = px.scatter_3d(df_comb, 
                       x='itr', 
                       y='dvfs', 
                       z='joule',
                       color='sys',
                       hover_data=['i', 'itr', 'rapl', 'dvfs', 'sys', 'instructions', 'c1', 'c7'],
                       custom_data=['i', 'itr', 'rapl', 'dvfs', 'sys', 'instructions', 'c1', 'c7'],
                       title='X=ITR Y=DVFS Z=JOULE',
                       width=800,
                       height=800)

        
app.layout = html.Div([
    dcc.Graph(
        id='edp-scatter',
        figure = edp_fig,
        style={'display': 'inline-block'},
    ),

    html.Div(id='my-output'),

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
        html.Hr(),
        html.Br()
    ]),

    dcc.Graph(
        id='custom-scatter3d',
        figure = td_fig,
        style={'display': 'inline-block'},
    ),
])

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
    Output(component_id='my-output', component_property='children'),
    Output('timeline-tx-bytes', 'figure'),
    Output('timeline-rx-bytes', 'figure'),
    Output('timeline-joules_diff', 'figure'),
    Output('timeline-timestamp_diff', 'figure'),
#    Output('timeline-c1_diff', 'figure'),
#    Output('timeline-c1e_diff', 'figure'),
#    Output('timeline-c3_diff', 'figure'),
#    Output('timeline-c7_diff', 'figure'),
    Output('timeline-nonidle_diff', 'figure'),
    Output('timeline-test1', 'figure'),
    Output('timeline-test2', 'figure'),
    Output('timeline-instructions_diff', 'figure'),
    Output('timeline-timestamp_diff_pointcloud', 'figure'),
    Output('timeline-joules_diff_pointcloud', 'figure'),
    [Input('edp-scatter', 'clickData')]
)
def update_timeline_lots(clickData):
    custom_data = clickData['points'][0]['customdata']
    #num, itr, rapl, dvfs, sys, c1, c1e, c3, c6, c7 = custom_data
    num, itr, rapl, dvfs, sys, instructions, c1, c7 = custom_data
    # custom_data = [1, 135, '0xffff', 'linux default', 'linux']

    global global_linux_default_df
    global global_linux_tuned_df
    global global_ebbrt_tuned_df
    global global_linux_default_df_non0j
    global global_linux_tuned_df_non0j
    global global_ebbrt_tuned_df_non0j    
    global global_linux_default_name
    global global_linux_tuned_name
    global global_ebbrt_tuned_name
    
    fname=''
    START_RDTSC=0
    END_RDTSC=0
    num_interrupts = 0
    
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

            num_interrupts = global_linux_tuned_df.shape[0]
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

            num_interrupts = global_linux_default_df.shape[0]
             
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
        fname = f'{log_loc}/ebbrt_dmesg.{num}_1_{itr}_{dvfs}_{rapl}.csv'
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
        global_ebbrt_tuned_df = pd.read_csv(fname, sep=' ', names=EBBRT_COLS, skiprows=1)
        global_ebbrt_tuned_df = global_ebbrt_tuned_df[global_ebbrt_tuned_df['timestamp'] >= START_RDTSC]
        global_ebbrt_tuned_df = global_ebbrt_tuned_df[global_ebbrt_tuned_df['timestamp'] <= END_RDTSC]
        global_ebbrt_tuned_df['timestamp'] = global_ebbrt_tuned_df['timestamp'] - global_ebbrt_tuned_df['timestamp'].min()
        global_ebbrt_tuned_df['timestamp'] = global_ebbrt_tuned_df['timestamp'] * TIME_CONVERSION_khz    
        global_ebbrt_tuned_df['joules'] = global_ebbrt_tuned_df['joules'] - global_ebbrt_tuned_df['joules'].min()
        global_ebbrt_tuned_df['joules'] = global_ebbrt_tuned_df['joules'] * JOULE_CONVERSION
        num_interrupts = global_ebbrt_tuned_df.shape[0]
         
        for c in ['instructions', 'ref_cycles', 'cycles', 'llc_miss', 'joules', 'timestamp', 'c3', 'c6', 'c7']: 
            global_ebbrt_tuned_df[c] = global_ebbrt_tuned_df[c] - global_ebbrt_tuned_df[c].min()        
        
        global_ebbrt_tuned_df_non0j = global_ebbrt_tuned_df[global_ebbrt_tuned_df['joules'] > 0].copy()
        global_ebbrt_tuned_df['timestamp_diff'] = global_ebbrt_tuned_df['timestamp'].diff()
        global_ebbrt_tuned_df.dropna(inplace=True)
        for c in ['instructions', 'ref_cycles', 'cycles', 'llc_miss', 'joules', 'timestamp', 'c3', 'c6', 'c7']:
            global_ebbrt_tuned_df_non0j[c] = global_ebbrt_tuned_df_non0j[c] - global_ebbrt_tuned_df_non0j[c].min()
        tmp = global_ebbrt_tuned_df_non0j[['instructions', 'ref_cycles', 'cycles', 'joules', 'timestamp', 'i', 'c3', 'c6', 'c7']].diff()
        tmp.columns = [f'{c}_diff' for c in tmp.columns]
        global_ebbrt_tuned_df_non0j = pd.concat([global_ebbrt_tuned_df_non0j, tmp], axis=1)        
        global_ebbrt_tuned_df_non0j['c1_diff'] = 0
        global_ebbrt_tuned_df_non0j['c1e_diff'] = 0
        global_ebbrt_tuned_df_non0j['nonidle_timestamp_diff'] = global_ebbrt_tuned_df_non0j['ref_cycles_diff'] * TIME_CONVERSION_khz
        global_ebbrt_tuned_df_non0j.dropna(inplace=True)
        global_ebbrt_tuned_df_non0j['nonidle_frac_diff'] = global_ebbrt_tuned_df_non0j['nonidle_timestamp_diff'] / global_ebbrt_tuned_df_non0j['timestamp_diff']
        global_ebbrt_tuned_name = [sys, num, itr, rapl, dvfs]
        
    fig1 = getFig('tx_bytes')
    fig2 = getFig('rx_bytes')
    fig3 = getFig('joules_diff', non0j=True)
    fig4 = getFig('timestamp_diff')
    fig10 = getFig('ref_cycles_diff', non0j=True)
    #fig5 = getFig('c1_diff', non0j=True)
    #fig6 = getFig('c1e_diff', non0j=True)
    #fig7 = getFig('c3_diff', non0j=True)
    #fig8 = getFig('c7_diff', non0j=True)
    fig9 = getFig('nonidle_frac_diff', non0j=True)
    fig11 = getFig('instructions_diff', non0j=True)
    fig12 = getFig('timestamp_diff', scatter=True)
    fig13 = getFig('joules_diff', non0j=True, scatter=True)
    
    #return f'Selected i={num} ITR-Delay={itr} RAPL={rapl} DVFS={dvfs} SYSTEM={sys}', fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10
    fig_test = go.Figure()
    fig_test.update_layout(xaxis_title="timestamp_diff (s)", yaxis_title=f'ref_cycles_diff')
    fig_test.add_trace(go.Pointcloud(x=global_ebbrt_tuned_df_non0j['timestamp_diff'], y=global_ebbrt_tuned_df_non0j['ref_cycles_diff'], name=f'{global_ebbrt_tuned_name}', showlegend=True, marker={'sizemin':2, 'color':'blue'}))
    return f'Selected i={num} ITR-Delay={itr} RAPL={rapl} DVFS={dvfs} SYSTEM={sys} NUM_INTERRUPTS={num_interrupts}', fig1, fig2, fig3, fig4, fig9, fig10, fig_test, fig11, fig12, fig13

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

if __name__=='__main__':
    app.run_server(host='10.241.31.7', port='8040')
#    app.run_server(debug=True)

'''
    dcc.Graph(
        id='timeline-c1_diff',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='timeline-c1e_diff',
        style={'display': 'inline-block'},
    ),

    dcc.Graph(
        id='timeline-c3_diff',
        style={'display': 'inline-block'},
    ),
    
    dcc.Graph(
        id='timeline-c7_diff',
        style={'display': 'inline-block'},
    ),
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
'''
