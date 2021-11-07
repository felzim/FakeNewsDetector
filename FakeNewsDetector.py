import pandas as pd

from src import config
from src import feature_processing
from src import train

import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc



# Initiale Daten laden und Modell trainieren
df = pd.read_csv(config.PROCESSED_DATA)
train.train_model(df)



def run_dash():

    app = dash.Dash(__name__)

    app.layout = html.Div([

        html.H1('Hello, I\'m a very simple Fake News Detector', style={'textAlign': 'center'}),

        html.Label('Bitte einen Datensatz auswählen '),
        dcc.Dropdown(
            id='demo-dropdown',
            options=[
                {'label': 'Datensatz 1', 'value': 'd1'},
                {'label': 'Datensatz 2', 'value': 'd2'},
                {'label': 'Datensatz 3', 'value': 'd3'}
            ],
            value='d1'
        ),

        dcc.Textarea(
            id='textarea-state-example',
            value='Hier Text eingeben',
            style={'width': '100%', 'height': 200},
        ),

        html.P(),
        html.Button('Prüfen', id='textarea-state-example-button', n_clicks=0),
        html.Div(id='textarea-state-example-output', style={'whiteSpace': 'pre-line'})
    ])

    @app.callback(

        Output('textarea-state-example-output', 'children'),
        Input('textarea-state-example-button', 'n_clicks'),
        State('textarea-state-example', 'value')
    )
    def update_output(n_clicks, value):

        res = train.check_fake_news(value)

        if n_clicks > 0:
            return f'\n\n Eingegebener Text: {format(value)}' \
                   f'\n\n Klassifiziert als: {res}' \
                   f'\n\n Analyse:'

    @app.callback(

        Output('dd-out put-container', 'children'),
        Input('demo-dropdown', 'value'),
    )
    def update_data(childen):
        return 'You have selected "{}"'.format(childen)


    app.run_server(debug=True)



# Dashboard starten
if __name__ == '__main__':
    run_dash()