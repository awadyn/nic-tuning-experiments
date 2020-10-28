import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import os

from config import *
from read_agg_data import *
from process_log_data import *

'''
Netpipe:
    Add all plots

Other workloads:
    Hook upto data

Layout etc.

Deployment:
    Web server on kd
'''

#Home-spun caching
agg_data = {}
log_data = {}

app = dash.Dash()

app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id = 'workload-selector',
            options = [
                {'label': 'Netpipe', 'value': 'netpipe'},
                {'label': 'NodeJS', 'value': 'nodejs'},
                {'label': 'Memcached', 'value': 'mcd'},
                {'label': 'Memcached Silo', 'value': 'mcdsilo'},
            ],
            value = 'netpipe'
        ),

        #TODO: combine netpipe and mcdsilo selectors. the should be populated by workload but keep one element
        dcc.RadioItems(
            id = 'netpipe-msg-selector',
            value = None
        ),

        dcc.RadioItems(
            id = 'mcdsilo-qps-selector',
            value = None
        ),

        dcc.RadioItems(
            id = 'aggregate-error-bar-selector',
            options = [{'label': 'On', 'value': 'on'}, {'label': 'Off', 'value': 'off'}],
            value = 'on'
        ),

        dcc.Graph(
            id='aggregate-scatter',
            style={'display': 'inline-block'},
        ),
    ],
    #style={'width': '49%', 'display': 'inline-block'}),
    ),

    #html.Div([
    #    dash_table.DataTable(id='aggregate-table')
    #]),

    #TODO: should be updated dynamically
    html.Div([
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

    ])

])

@app.callback(
    Output('netpipe-msg-selector', 'options'),
    Output('netpipe-msg-selector', 'value'),
    [Input('workload-selector', 'value')]
)
def update_radio_button(workload):
    if workload=='netpipe':
        options = [{'label': key, 'value': key} for key in netpipe_msg_sizes]
        value = 8192

    elif workload=='mcdsilo':
        options = [{'label': key, 'value': key} for key in mcd_qps_sizes]
        value = 200000

    else:
        options = []
        value = None

    return options, value

@app.callback(
    Output('aggregate-scatter', 'figure'),
    #Output('aggregate-table', 'data'),
    #Output('aggregate-table', 'columns'),
    [Input('workload-selector', 'value'),
     Input('netpipe-msg-selector', 'value'),
     Input('aggregate-error-bar-selector', 'value')]
    )
def update_aggregate_plot(workload, msg, agg_err_bar):
    #TODO: fix start_analysis for other workloads
    if workload in agg_data:
        df_comb, df, outlier_list = agg_data[workload]
    else:
        df_comb, df, outlier_list = start_analysis(workload=workload)
        df_comb.reset_index(inplace=True)
        
        #TODO: lower dvfs for non-netpipe workloads
        df_comb['Sys'] = df_comb.apply(lambda x: x['sys'] + ' default' if x['sys']=='linux' and x['dvfs'].lower()=='0xffff' and x['itr']==1 else x['sys'], axis=1)

        agg_data[workload] = (df_comb, df, outlier_list)
    
    if workload=='netpipe':
        df_comb = df_comb[df_comb['msg']==msg]
    elif workload=='mcdsilo':
        df_comb = df_comb[df_comb['QPS']==msg]

    #TODO: look at update methods instead of having if-else block with two calls to px.scatter
    #TODO: check Sys column
    #TODO: add number of runs

    if agg_err_bar=='on':
        fig = px.scatter(df_comb, 
                         x='time_mean', 
                         y='joules_mean', 
                         error_x='time_std', 
                         error_y='joules_std', 
                         color='Sys',
                         labels={'time_mean': 'Time (s)', 'joules_mean': 'Energy (Joules)'}, 
                         hover_data=['itr', 'rapl', 'dvfs', 'Sys'], 
                         custom_data=['itr', 'rapl', 'dvfs', 'Sys', 'sys'],
                         title=f'Netpipe {msg} bytes Global Plot')

    else:
        fig = px.scatter(df_comb, 
                         x='time_mean', 
                         y='joules_mean', 
                         color='Sys',
                         labels={'time_mean': 'Time (s)', 'joules_mean': 'Energy (Joules)'}, 
                         hover_data=['itr', 'rapl', 'dvfs', 'Sys'], 
                         custom_data=['itr', 'rapl', 'dvfs', 'Sys', 'sys'],
                         title=f'Netpipe {msg} bytes Global Plot')


    return fig #, df_comb.to_dict('records'), [{'name': i, 'id': i} for i in df_comb.columns]

#TODO: don't hardcode number of outputs. should be dynamic based on len(timeline_plot_metrics)
@app.callback(
    Output('timeline-tx-bytes', 'figure'),
    Output('timeline-rx-bytes', 'figure'),
    Output('timeline-joules_diff', 'figure'),
    Output('timeline-timestamp_diff', 'figure'),

    [Input('workload-selector', 'value'),
     Input('netpipe-msg-selector', 'value'),
     Input('aggregate-scatter', 'clickData')], #TODO: this shouldn't be netpipe specific
)
def update_timeline_plots(workload, msg, clickdata):
    #construct filename
    if workload=='netpipe':
        #filename = os.path.join(Locations.netpipe_logs_loc,)
        #filename = '/home/sanjay/sesa-papers/ukpwr/logs/aug19_netpipelogs/governor/linux.dmesg.7_65536_5000_1_0xFFFF_135.csv'

        #TODO: check if we can have a clickdata default in Graph
        if clickdata:
            custom_data = clickdata['points'][0]['customdata']
        else:
            custom_data = [1, 135, '0xFFFF', 'linux default', 'linux']

        assert(len(custom_data)==5)
        itr, rapl, dvfs, _, sys = custom_data

        if sys=='linux' and itr==1 and dvfs.lower()=='0xffff': 
            folder = 'governor'
        else: 
            folder = 'rapl135'

        #TODO: add selector for run number (the "1" after dmesg)
        filename = os.path.join(Locations.netpipe_logs_loc, folder, f'{sys}.dmesg.1_{msg}_5000_{itr}_{dvfs}_{rapl}.csv')    
        print('Netpipe', filename)    
        
        skiprows = 0
        pass_colnames = False

    elif workload=='nodejs':

        #TODO: check if we can have a clickdata default in Graph
        if clickdata:
            custom_data = clickdata['points'][0]['customdata']
        else:
            custom_data = [1, 135, '0xffff', 'linux default', 'linux']

        assert(len(custom_data)==5)
        itr, rapl, dvfs, _, sys = custom_data

        ostag = 'node' if sys=='linux' else 'ebbrt'
        extension = '.csv' if sys=='ebbrt' else ''
        filename = os.path.join(Locations.nodejs_logs_loc, f'{ostag}_dmesg.1_1_{itr}_{dvfs}_{rapl}{extension}')

        print('NodeJS', filename)

        skiprows = 1
        pass_colnames = True

    #read log file
    df, df_orig = process_log_file(filename, ts_filename=None, skiprows=skiprows, pass_colnames=pass_colnames)

    print(df.columns)
    print(df.head())

    #make plots
    #TODO: add selector for runs
    fig_list = []
    for m in PlotList.timeline_plots_metrics:
        if m.find('tx')==0 or m.find('rx')==0:
            df_to_use = df_orig
        else:
            df_to_use = df

        fig = px.scatter(df_to_use,
                         x='timestamp',
                         y=m,
                         labels={'timestamp': 'Time(s)', m: m},
                         title=f'Timeline for metric = {m}')
        fig_list.append(fig)

    return tuple(fig_list)

'''
Plots:

Surfaces
Energy timeline
tx_bytes/rx_bytes timelines
Counters (N_interrupts etc.)

Postpone:
Comparison between two selections

'''

if __name__=='__main__':
    app.run_server(debug=True)