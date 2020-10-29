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

        dcc.RadioItems(
            id = 'msg-qps-selector',
            value = None
        ),

        dcc.RadioItems(
            id = 'aggregate-error-bar-selector',
            options = [{'label': 'On', 'value': 'on'}, {'label': 'Off', 'value': 'off'}],
            value = 'on'
        ),

        dcc.Dropdown(id='xaxis-selector', value='time_mean'),
        dcc.Dropdown(id='yaxis-selector', value='joules_mean'),

    ],
    ),

    #TODO: should be updated dynamically
    html.Div([
        dcc.Graph(
            id='edp-scatter',
            style={'display': 'inline-block'},
        ),

        dcc.Graph(
            id='custom-scatter',
            style={'display': 'inline-block'},
        ),

    ]),

    html.Div([
        dcc.Dropdown(id='xaxis3d-selector', value='time_mean'),
        dcc.Dropdown(id='yaxis3d-selector', value='joules_mean'),
        dcc.Dropdown(id='zaxis3d-selector', value='itr'),

        dcc.Graph(
            id='custom-scatter3d',
            style={'display': 'inline-block'},
        ),

    ])


])

@app.callback(
    Output('msg-qps-selector', 'options'),
    Output('msg-qps-selector', 'value'),
    [Input('workload-selector', 'value')]
)
def update_radio_button(workload):
    options = [{'label': key, 'value': key} for key in msg_qps_sizes[workload]['options']]
    value = msg_qps_sizes[workload]['value']

    return options, value

@app.callback(
    Output('edp-scatter', 'figure'),
    Output('xaxis-selector', 'options'),
    Output('yaxis-selector', 'options'),

    Output('xaxis3d-selector', 'options'),
    Output('yaxis3d-selector', 'options'),
    Output('zaxis3d-selector', 'options'),

    [Input('workload-selector', 'value'), 
     Input('msg-qps-selector', 'value'), 
     Input('aggregate-error-bar-selector', 'value')]
)
def update_edp_scatter(workload, msg, agg_err_bar):
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

    elif workload=='mcd' or workload=='mcdsilo':
        df_comb = df_comb[df_comb['QPS']==msg]

    xaxis_values = [{'label': key, 'value': key} for key in df_comb.columns]
    yaxis_values = [{'label': key, 'value': key} for key in df_comb.columns]

    if agg_err_bar=='on':
        fig = px.scatter(df_comb, 
                         x='time_mean', 
                         y='joules_mean', 
                         error_x='time_std', 
                         error_y='joules_std', 
                         color='Sys',
                         labels={'time_mean': 'Time (s)', 'joules_mean': 'Energy (Joules)'}, 
                         hover_data=hover_data[workload],
                         custom_data=['itr', 'rapl', 'dvfs', 'Sys', 'sys'],
                         title=f'{workload.capitalize()} {msg} bytes Global Plot')


    else:
        fig = px.scatter(df_comb, 
                         x='time_mean', 
                         y='joules_mean', 
                         color='Sys',
                         labels={'time_mean': 'Time (s)', 'joules_mean': 'Energy (Joules)'}, 
                         hover_data=hover_data[workload],
                         custom_data=['itr', 'rapl', 'dvfs', 'Sys', 'sys', 'instructions_mean'],
                         title=f'{workload.capitalize()} {msg} bytes Global Plot')


    return fig, xaxis_values, yaxis_values, yaxis_values, yaxis_values, yaxis_values #, df_comb.to_dict('records'), [{'name': i, 'id': i} for i in df_comb.columns]

@app.callback(
    Output('custom-scatter', 'figure'),
    [Input('workload-selector', 'value'), 
     Input('msg-qps-selector', 'value'), 
     Input('aggregate-error-bar-selector', 'value'),
     Input('xaxis-selector', 'value'),
     Input('yaxis-selector', 'value')]
)
def update_custom_plot(workload, msg, agg_err_bar, xcol, ycol):
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

    elif workload=='mcd' or workload=='mcdsilo':
        df_comb = df_comb[df_comb['QPS']==msg]

    if agg_err_bar=='on':
        fig = px.scatter(df_comb, 
                         x=xcol, 
                         y=ycol, 
                         error_x=xcol.replace('_mean', '_std'), 
                         error_y=ycol.replace('_mean', '_std'), 
                         color='Sys',
                         hover_data=hover_data[workload],
                         custom_data=['itr', 'rapl', 'dvfs', 'Sys', 'sys'],
                         title=f'{workload.capitalize()} {msg} bytes Global Plo')


    else:
        fig = px.scatter(df_comb, 
                         x=xcol, 
                         y=ycol, 
                         color='Sys',
                         hover_data=hover_data[workload],
                         custom_data=['itr', 'rapl', 'dvfs', 'Sys', 'sys'],
                         title=f'{workload.capitalize()} {msg} bytes Global Plo')

    return fig

@app.callback(
    Output('custom-scatter3d', 'figure'),
    [Input('workload-selector', 'value'), 
     Input('msg-qps-selector', 'value'), 
     Input('aggregate-error-bar-selector', 'value'),     
     Input('xaxis3d-selector', 'value'),
     Input('yaxis3d-selector', 'value'),
     Input('zaxis3d-selector', 'value')]
)
def update_custom_plot3d(workload, msg, agg_err_bar, xcol, ycol, zcol):
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

    elif workload=='mcd' or workload=='mcdsilo':
        df_comb = df_comb[df_comb['QPS']==msg]

    if agg_err_bar=='on':
        fig = px.scatter_3d(df_comb, 
                            x=xcol, 
                            y=ycol, 
                            z=zcol,
                            error_x=xcol.replace('_mean', '_std'), 
                            error_y=ycol.replace('_mean', '_std'), 
                            error_z=zcol.replace('_mean', '_std'),
                            color='Sys',
                            size='edp_mean',
                            hover_data=hover_data[workload],
                            custom_data=['itr', 'rapl', 'dvfs', 'Sys', 'sys'],
                            title=f'{workload.capitalize()} {msg} bytes Global Plo',
                            width=800,
                            height=800)

    else:
        fig = px.scatter_3d(df_comb, 
                            x=xcol, 
                            y=ycol, 
                            z=zcol,
                            color='Sys',
                            size='edp_mean',
                            hover_data=hover_data[workload],
                            custom_data=['itr', 'rapl', 'dvfs', 'Sys', 'sys'],
                            title=f'{workload.capitalize()} {msg} bytes Global Plo',
                            width=800,
                            height=800)

    return fig


if __name__=='__main__':
    app.run_server(debug=True)
    #app.run_server(host='10.241.31.7', port='8050')