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
        dcc.Dropdown(id='sys-selector', value='linux'),
        dcc.Dropdown(id='itr-selector', value=1),
        dcc.Dropdown(id='dvfs-selector', value='0xFFFF'),
        dcc.Dropdown(id='rapl-selector', value=135),
        dcc.Dropdown(id='df-selector', value='Interrupts', options=[{'label': 'Interrupts', 'value': 'Interrupts'}, {'label': 'Energy', 'value': 'Energy'}]),
        dcc.Dropdown(id='xaxis-selector', value='timestamp'),
        dcc.Dropdown(id='yaxis-selector', value='tx_bytes'),

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
    ),

    #TODO: should be updated dynamically
    html.Div([
        dcc.Graph(
            id='custom-scatter',
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

    elif workload=='mcd' or workload=='mcdsilo':
        options = [{'label': key, 'value': key} for key in mcd_qps_sizes]
        value = 200000

    else:
        options = []
        value = None

    return options, value

@app.callback(
    Output('aggregate-scatter', 'figure'),
    Output('sys-selector', 'options'),
    Output('itr-selector', 'options'),
    Output('dvfs-selector', 'options'),
    Output('rapl-selector', 'options'),
    Output('xaxis-selector', 'options'),
    Output('yaxis-selector', 'options'),

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
                         title=f'{workload.capitalize()} {msg} bytes Global Plot')

    else:
        fig = px.scatter(df_comb, 
                         x='time_mean', 
                         y='joules_mean', 
                         color='Sys',
                         labels={'time_mean': 'Time (s)', 'joules_mean': 'Energy (Joules)'}, 
                         hover_data=['itr', 'rapl', 'dvfs', 'Sys'], 
                         custom_data=['itr', 'rapl', 'dvfs', 'Sys', 'sys'],
                         title=f'{workload.capitalize()} {msg} bytes Global Plot')


    Output('sys-selector', 'value'),
    Output('itr-selector', 'value'),
    Output('dvfs-selector', 'value'),
    Output('rapl-selector', 'value'),

    sys_selector = [{'label': k, 'value': k} for k in df_comb['sys'].unique()]
    itr_selector = [{'label': k, 'value': k} for k in df_comb['itr'].unique()]
    dvfs_selector = [{'label': k, 'value': k} for k in df_comb['dvfs'].unique()]
    rapl_selector = [{'label': k, 'value': k} for k in df_comb['rapl'].unique()]

    xaxis_selector = yaxis_selector = [{'label': k, 'value': k} for k in COLS]

    return fig, sys_selector, itr_selector, dvfs_selector, rapl_selector, xaxis_selector, yaxis_selector #, df_comb.to_dict('records'), [{'name': i, 'id': i} for i in df_comb.columns]

#TODO: don't hardcode number of outputs. should be dynamic based on len(timeline_plot_metrics)
@app.callback(
    Output('custom-scatter', 'figure'),
    [Input('workload-selector', 'value'),
     Input('netpipe-msg-selector', 'value'),
     Input('sys-selector', 'value'),
     Input('itr-selector', 'value'),
     Input('dvfs-selector', 'value'),
     Input('rapl-selector', 'value'),
     Input('df-selector', 'value'),
     Input('xaxis-selector', 'value'),
     Input('yaxis-selector', 'value'),
    ])
def update_logfile_plots(workload,
                         msg,
                         sys,
                         itr,
                         dvfs,
                         rapl,
                         df_selector,
                         xaxis,
                         yaxis):

    #construct filename
    if workload=='netpipe':
        #TODO: add selector for run number (the "1" after dmesg)
        if sys=='linux':
            filename = os.path.join(Locations.netpipe_logs_loc, Locations.netpipe_linux_subfolder, f'dmesg_devicelog.0_{msg}_5000_{itr}_{dvfs}_{rapl}')
            ts_filename = None
            pass_colnames = False
        
        elif sys=='ebbrt':
            filename = os.path.join(Locations.netpipe_logs_loc, Locations.netpipe_ebbrt_subfolder, f'ebbrt.dmesg.5_{msg}_5000_{itr}_{dvfs}_{rapl}')
            ts_filename = None
            pass_colnames = True

        skiprows = 1
        ts_start_idx = None
        ts_end_idx = None
    
        print('Netpipe', filename)    

    elif workload=='nodejs':
        #TODO: add selector for run number (the "1" after dmesg)
        if sys=='linux':
            filename = os.path.join(Locations.nodejs_logs_loc, Locations.nodejs_linux_subfolder, f'node_dmesg.9_1_{itr}_{dvfs}_{rapl}')
            ts_filename = os.path.join(Locations.nodejs_logs_loc, Locations.nodejs_linux_subfolder, f'node_rdtsc.9_1_{itr}_{dvfs}_{rapl}')
            pass_colnames = False
            skiprows = 1
            ts_start_idx = 1
            ts_end_idx = 2
        
        elif sys=='ebbrt':
            filename = os.path.join(Locations.nodejs_logs_loc, Locations.nodejs_ebbrt_subfolder, f'ebbrt_dmesg.9_1_{itr}_{dvfs}_{rapl}.csv')
            ts_filename = os.path.join(Locations.nodejs_logs_loc, Locations.nodejs_ebbrt_subfolder, f'ebbrt_rdtsc.9_{itr}_{dvfs}_{rapl}')
            pass_colnames = True
            skiprows = 1
            ts_start_idx = 0
            ts_end_idx = 1
    
        print('NodeJS', filename, ts_filename)

    elif workload=='mcd':
        #TODO: add selector for run number (the "1" after dmesg)
        if sys=='linux':
            filename = os.path.join(Locations.mcd_logs_loc, Locations.mcd_linux_subfolder, f'mcd_dmesg.0_4_{itr}_{dvfs}_{rapl}_{qps}')
            ts_filename = os.path.join(Locations.mcd_logs_loc, Locations.mcd_linux_subfolder, f'mcd_rdtsc.4_{itr}_{dvfs}_{rapl}_{qps}')
            pass_colnames = False
            skiprows = 1
            ts_start_idx = 2
            ts_end_idx = 3
        
        elif sys=='ebbrt':
            filename = os.path.join(Locations.mcd_logs_loc, Locations.mcd_ebbrt_subfolder, f'ebbrt_dmesg.0_4_{itr}_{dvfs}_{rapl}_{qps}.csv')
            ts_filename = os.path.join(Locations.mcd_logs_loc, Locations.mcd_ebbrt_subfolder, f'ebbrt_rdtsc.9_{itr}_{dvfs}_{rapl}_{qps}')
            pass_colnames = True
            skiprows = 1
            ts_start_idx = 0
            ts_end_idx = 1
    
        print('Memcached', filename, ts_filename)

    elif workload=='mcdsilo':
        #TODO: add selector for run number (the "1" after dmesg)
        if sys=='linux':
            filename = os.path.join(Locations.mcdsilo_logs_loc, Locations.mcdsilo_linux_subfolder, f'node_dmesg.9_1_{itr}_{dvfs}_{rapl}_{qps}')
            ts_filename = os.path.join(Locations.mcdsilo_logs_loc, Locations.mcdsilo_linux_subfolder, f'node_rdtsc.9_1_{itr}_{dvfs}_{rapl}_{qps}')
            pass_colnames = False
            skiprows = 1
            ts_start_idx = 1
            ts_end_idx = 2
        
        elif sys=='ebbrt':
            filename = os.path.join(Locations.mcdsilo_logs_loc, Locations.mcdsilo_ebbrt_subfolder, f'ebbrt_dmesg.9_1_{itr}_{dvfs}_{rapl}_{qps}.csv')
            ts_filename = os.path.join(Locations.mcdsilo_logs_loc, Locations.mcdsilo_ebbrt_subfolder, f'ebbrt_rdtsc.9_{itr}_{dvfs}_{rapl}_{qps}')
            pass_colnames = True
            skiprows = 1
            ts_start_idx = 0
            ts_end_idx = 1
    
        print('Memcached Silo', filename, ts_filename)

    #read log file
    #TODO: time scaling for nodejs logs
    df, df_orig = process_log_file(filename, 
                                   ts_filename=ts_filename, 
                                   ts_start_idx=ts_start_idx,
                                   ts_end_idx=ts_end_idx,
                                   pass_colnames=pass_colnames, 
                                   skiprows=skiprows)

    fig = px.scatter(df_orig, 
                     x=xaxis, 
                     y=yaxis)

    return fig

if __name__=='__main__':
    app.run_server(debug=True)
    #app.run_server(host='10.241.31.7', port='8050')