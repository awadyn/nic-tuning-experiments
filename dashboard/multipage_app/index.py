import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

from app import app
from apps import app1, app2, node, mcd200K

print('dcc version = ', dcc.__version__) # 0.6.0 or above is required

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    dcc.Link('Go to App 1', href='/apps/app1'),
    html.Br(),
    dcc.Link('Go to App 2', href='/apps/app2'),
    html.Br(),
    dcc.Link('Go to NodeJS', href='/apps/node'),
    html.Br(),
    dcc.Link('Go to Memcached QPS 200K', href='/apps/mcd200K')    
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    print(pathname)
    if pathname == '/apps/app1':
        return app1.layout
    elif pathname == '/apps/app2':
        return app2.layout
    elif pathname == '/apps/node':
        return node.layout
    elif pathname == '/apps/mcd200K':
        return mcd200K.layout

if __name__ == '__main__':
    app.run_server(host='10.241.31.7', port='8030', debug=True)
