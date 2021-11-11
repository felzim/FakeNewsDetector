import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from src import config
from src import feature_processing
from src import train_and_check

import dash
from dash import callback_context
from dash.dependencies import Input, Output, State
from dash import html
from dash import dcc


####################### Simple Fake News Detector 0.1
########## Autor: Felix Zimmermann
#
#    Diese sehr einfache Demo baut auf einem Datensatz von kaggle.com auf, das insgesamt 17.903 als
#    Fake-News und 20.826 als True-News gelabelte Nachrichten enthält.
#    (Link: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
#
#    Ziel ist es, einen beliebigen, englischsprachigen Text entweder als "Fake-News"
#    oder als "Real News" zu klassifizieren. Hierzu wurde der Datensatz zunächst bereinigt
#    (feature_processing.py). Anschließend werden mit Hilfe des Tfidf-Vectorizers aus Sklearn die
#    unstrukturierten in numerische Daten überführt und mit dem PassiveAggressiveClassifier ein
#    predict ausgeführt.



# Den bereits bereinigten Datensatz laden
df = pd.read_csv(config.PROCESSED_DATA)

# Split in Trainings- und Testdaten
X = df['text']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Modell trainieren
train_and_check.train_model(X_train, y_train)



# Web-Interface auf 127.0.0.1:8050
def run_dash():

    app = dash.Dash(__name__)

    app.layout = html.Div([

        html.Link(
            rel='stylesheet',
            href='assets/custom_loader.css'
        ),

        html.H1('Hello, I\'m a very simple Fake News Detector', style={'textAlign': 'center'}),

        dcc.Textarea(
            id='textarea-status',
            value='True / Fake?',
            style={'width': '50%', 'height': 200},
        ),


        html.P(),
        html.Button('Text prüfen', id='text-pruefen-button', n_clicks=0),
        html.Div(id='textarea-status-output', style={'whiteSpace': 'pre-line',
                                                     'font-size': '0.9em',
                                                     'height': 150}),

        html.Span("Sonstiges: "),
        html.Button('Feature processing ausführen', id='processing-button', n_clicks=0),
        html.P(),

    ])

    @app.callback(

        Output('textarea-status-output', 'children'),

        [Input('text-pruefen-button', 'n_clicks'),
         Input('processing-button', 'n_clicks')],
        State('textarea-status', 'value')
    )
    def update_output(btn1, btn2, value):

        if len(value) > 0:
            res = train_and_check.detect_fake_news(value)
            overlap = train_and_check.overlap_training_data(value, X_train)

            changed_id = [p['prop_id'] for p in callback_context.triggered][0]

            if 'text-pruefen-button' in changed_id:
                return  f'\n\n Klassifiziert als: {res}' \
                        f'\n\n Overlap: {overlap}% der eingegebenen Wörter sind in den Trainingsdaten enthalten.'

            elif 'processing-button' in changed_id:
                feature_processing.process()
                return f'\n\n Feature processing abgeschlossen.'



    app.run_server(debug=True)


# Dashboard starten
if __name__ == '__main__':
    run_dash()