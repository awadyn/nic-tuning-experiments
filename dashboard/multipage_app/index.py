import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

from app import app
from apps import app1, app2, node, mcd200K, mcd400K, mcd600K, mcdsilo50K, mcdsilo100K, mcdsilo200K

print('dcc version = ', dcc.__version__) # 0.6.0 or above is required

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    dcc.Link('Test', href='/apps/app1'),
    html.Br(),
    dcc.Link('Go to NodeJS', href='/apps/node'),
    html.Br(),
    dcc.Link('Go to Memcached QPS 200K', href='/apps/mcd200K'),
    html.Br(),
    dcc.Link('Go to Memcached QPS 400K', href='/apps/mcd400K'),
    html.Br(),
    dcc.Link('Go to Memcached QPS 600K', href='/apps/mcd600K'),
    html.Br(),
    dcc.Link('Go to Memcached-Silo QPS 50K (Under construction)', href='/apps/mcdsilo50K'),
    html.Br(),
    dcc.Link('Go to Memcached-Silo QPS 100K (Under construction)', href='/apps/mcdsilo100K'),
    html.Br(),
    dcc.Link('Go to Memcached-Silo QPS 200K', href='/apps/mcdsilo200K')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    print(pathname)
    if pathname == '/apps/node':
        return node.layout
    elif pathname == '/apps/mcd200K':
        return mcd200K.layout
    elif pathname == '/apps/mcd400K':
        return mcd400K.layout
    elif pathname == '/apps/mcd600K':
        return mcd600K.layout
    elif pathname == '/apps/mcdsilo50K':
        return mcdsilo50K.layout
    elif pathname == '/apps/mcdsilo100K':
        return mcdsilo100K.layout
    elif pathname == '/apps/mcdsilo200K':
        return mcdsilo200K.layout
    elif pathname == '/apps/app1':
        return app1.layout    

if __name__ == '__main__':
    #app.run_server(host='10.241.31.7', port='8040', debug=True)
    app.run_server(host='10.241.31.7', port='8040')
