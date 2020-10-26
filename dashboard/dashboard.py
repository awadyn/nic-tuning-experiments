import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

from read_agg_data import *

#Home-spun caching
agg_data = {}

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
            id = 'netpipe-msg-selector',
            value = None
        ),

        dcc.RadioItems(
            id = 'aggregate-error-bars',
            options = [{'label': 'On', 'value': 'on'}, {'label': 'Off', 'value': 'off'}],
            value = 'on'
        ),

        dcc.Graph(
            id='aggregate-scatter',
            style={'display': 'inline-block'},
        ),
    ],
    style={'width': '49%', 'display': 'inline-block'}),

    html.Div([]),

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
    else:
        options = []
        value = None

    return options, value

@app.callback(
    Output('aggregate-scatter', 'figure'),
    [Input('workload-selector', 'value'),
     Input('netpipe-msg-selector', 'value')]
    )
def update_aggregate_plot(workload, msg):
    #TODO: fix start_analysis for other workloads
    if workload in agg_data:
        df_comb, df, outlier_list = agg_data[workload]
    else:
        df_comb, df, outlier_list = start_analysis(workload=workload)
        df_comb.reset_index(inplace=True)
        
        #TODO: lower dvfs for non-netpipe workloads
        df_comb['Sys'] = df_comb.apply(lambda x: x['sys'] + ' default' if x['sys']=='linux' and x['dvfs']=='0xFFFF' and x['itr']==1 else x['sys'], axis=1)

        agg_data[workload] = (df_comb, df, outlier_list)
    
    df_comb = df_comb[df_comb['msg']==msg]
    
    fig = px.scatter(df_comb, 
                     x='time_mean', 
                     y='joules_mean', 
                     error_x='time_std', 
                     error_y='joules_std', 
                     color='Sys',
                     labels={'time_mean': 'Time (s)', 'joules_mean': 'Energy (Joules)'}, 
                     hover_data=['itr', 'rapl', 'dvfs'], title=f'Netpipe {msg} bytes Global Plot')

    return fig

if __name__=='__main__':
    app.run_server(debug=True)