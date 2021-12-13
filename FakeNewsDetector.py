from src import config

import pandas as pd
import numpy as np

# Reguläre Ausdrücke
import re

# Sklearn
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Balancieren / Undersampling
from imblearn.under_sampling import RandomUnderSampler

# Metriken
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Dash-Komponenten
import dash
from dash import callback_context
from dash.dependencies import Input, Output, State
from dash import html
from dash import dcc

# NLTK
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
stopwords = set(stopwords)


df_fake = pd.read_csv(config.DATA_FAKE)
df_true = pd.read_csv(config.DATA_TRUE)


####################### Simple Fake News Detector 0.1
########## Autor: Felix Zimmermann
#
#    Diese sehr einfache Demo baut auf einem Datensatz von kaggle.com auf, das insgesamt 17.903 als
#    Fake-News und 20.826 als True-News gelabelte Nachrichten enthält.
#    (Link: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
#
#    Ziel ist es, einen beliebigen, englischsprachigen Text entweder als "Fake-News"
#    oder als "Real News" zu klassifizieren.
#    TODO: Readme


class FakeNewsDetector():

    def __init__(self):

        self.last_result = 'init'
        self.accuracy_score = None
        self.precision_score = None
        self.recall_score = None
        self.f1_score = None

        # TFIDF Vectorizer initialisieren
        self.tfvect = TfidfVectorizer(stop_words='english', max_df=0.90)

        # PassiveAggressive Classifier initialisieren
        self.classifier = PassiveAggressiveClassifier()

        self.process()
        self.train_model()


    def train_model(self):

        # Train und Test mit fit_transform in eine Matrix konvertieren / vektorisieren
        self.tfidf_x_train = self.tfvect.fit_transform(self.X_train.astype('U').values)

        # Datensatz balancieren
        rus = RandomUnderSampler()
        self.tfidf_x_train_rs, self.y_train_rs = rus.fit_resample(self.tfidf_x_train, self.y_train)

        # Classifier trainieren
        self.classifier.fit(self.tfidf_x_train_rs, self.y_train_rs)

        # Score berechnen
        y_pred = self.classifier.predict(self.tfvect.transform(self.X_test))

        self.accuracy_score = round(accuracy_score(self.y_test, y_pred),2)
        self.precision_score = round(precision_score(self.y_test, y_pred),2)
        self.recall_score = round(recall_score(self.y_test, y_pred),2)
        self.f1_score = round(f1_score(self.y_test, y_pred),2)


    def detect_fake_news(self, news):
        ''' Diese Funktion gibt für einen beliebigen String eine eindeutige Klassifizierung, ob True oder Fake
        Dafür nutzt sie den zuvor trainierten Classifier.
        '''

        input_data = [news]

        # eingegebenen Text mit dem bereits gefitteten Modell in numerisches Format transformieren
        vectorized_input_data = self.tfvect.transform(input_data)
        prediction = self.classifier.predict(vectorized_input_data)[0]

        if prediction == 0:
            return ["Fake", self.classifier.decision_function(vectorized_input_data)[0]]
        if prediction == 1:
            return ["True", self.classifier.decision_function(vectorized_input_data)[0]]


    def overlap_training_data(self, news):
        ''' Funktion, die prüft, zu welchem Prozentsatz die Wörter eines Strings
        in den Trainingsdaten enthalten sind'''

        input_data = news
        list_of_words = []

        for word in input_data:
            list_of_words = input_data.split()

        # Sonderzeichen entfernen
        list_of_words = [re.sub('[^a-zA-Z0-9]+', '', word) for word in list_of_words]

        # Leere Elemente in der Liste aussortieren
        list_of_words = [x for x in list_of_words if x]

        count_enthalten = 0
        count_not_enthalten = 0


        # Zusammenzählen
        for word in list_of_words:
            res = set(self.X_train.str.contains(word.lower(), regex=False))
            if True in res:
                count_enthalten += 1
            else:
                count_not_enthalten += 1

        return round((count_enthalten / len(list_of_words) * 100), 2)


    def process(self, df_true_= df_true, df_fake_=df_fake, dropwords=[], test_size=0.25):
        ''' Funktion, die aus zwei Dataframes einen neuen zusammensetzt und diesen bereinigt
            Es kann optional eine Liste von Wörtern mit übergeben werden, die dann aus dem neuen DF herausgefiltert werden
        '''

        # vor dem Zusammensetzen das target festschreiben
        df_true_['category'] = 1
        df_fake_['category'] = 0

        # neuen Dataframe mit true and fake
        df_ = pd.concat([df_true_, df_fake_])

        # neuen Index erstellen und alten Index droppen. Titel-Spalte ebenfalls entfernen
        df_ = df_.reset_index(drop=False)
        df_ = df_.drop(columns=['index', 'title'])

        self.X_raw = df_['text']

        # Sonderzeichen entfernen (alles, was nicht a-zA-Z# ist)
        df_['text'] = df_['text'].str.replace("[^a-zA-Z#]", " ", regex=True)

        # Alle Wörter, die länger als 3 Zeichen sind, entfernen
        df_['text'] = df_['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

        # Alle Duplikate in der Textspalte droppen
        df_ = df_.drop_duplicates(subset=['text'])

        # Index neu erstellen nachdem Zeilen entfernt wurden
        df_ = df_.reset_index(drop=False)
        df_ = df_.drop(columns='index')

        # Textspalte in lowercase umschreiben
        df_['text'] = df_['text'].str.lower()

        # alle stopwords entfernen
        df_['text'] = df_['text'].apply(lambda x: ' '.join([w for w in x.split() if w not in stopwords]))

        # weitere Bereinigung
        df_['text'] = df_['text'].str.replace("reuters", '')
        df_['text'] = df_['text'].str.replace("washington", '')

        for word in dropwords:
            df_['text'] = df_['text'].str.replace(word, '')

        # bearbeitete Daten als CSV speichern
        df_.to_csv(config.PROCESSED_DATA)

        X = df_['text']
        y = df_['category']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)


    # Web-Interface auf 127.0.0.1:8050
    def run_dash(self):

        app = dash.Dash(__name__)

        app.layout = html.Div([


            html.Link(
                rel='stylesheet',
                href='assets/custom_loader.css'
            ),


            html.Div([

                    html.H2('A very simple Fake News Detector',
                        style={'font-family': 'Georgia'}),


                    html.P(
                        style={'height': '1em'}
                    ),


                    html.Button('Text prüfen', id='text-pruefen-button', n_clicks=0),

                    html.Br(),
                    dcc.Textarea(
                        id='textarea-status',
                        value=self.X_raw[np.random.randint(len(self.X_raw))],
                        style={'width': '100%','height': '200px', 'margin-top': '1em', 'padding': '12px', 'background-color': '#fbfbfb', 'font-family':'Calibri, sans-serif', 'color':'#353535'},
                    ),


                    html.Div([


                        html.Div(id='textarea-status-output',
                                 style={
                                'font-size': '0.9em'},
                        ),

                        html.Img(
                            id="res-image",
                            style={'width': '25px'},
                        ),

                    ],
                    style={'margin-top': '0.5em', 'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'align-items': 'flex-start', 'gap': '15px', 'height':'100px'}),


                    html.P(
                        style={'clear': 'both'}
                    ),

                    html.H4("Optionen: "),
                    html.Span("Features aus Datensatz entfernen: "),

                    html.Br(),

                    html.Div([

                        dcc.Input(
                                id='input-dropwords',
                                placeholder='Wörter durch Komma getrennt eingeben...',
                                type='text',
                                value='',
                                style={'width': '30%'}
                        ),
                        html.Button('Feature processing ausführen', id='processing-button', n_clicks=0),
                        html.Button('Modell neu berechnen', id='model-button', n_clicks=0),

                    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'flex-start', 'align-items': 'stretch', 'gap':'10px', 'margin-top':'1em'}),


                    html.Div(id='textarea-status-output2', style={
                            'white-space': 'pre-line',
                            'font-size': '0.9em',
                            'margin-top': '1em',
                            'height': 150},),




                ],
                style={'margin': '0 auto', 'text-align': 'left', 'width': '40%'}),


        ],
        style={'text-align': 'center'})



        @app.callback(
            Output("res-image", "src"),

            Input('textarea-status-output', 'children'),
        )
        def update_res_image(textarea_value):

            if self.last_result == 'True':
                src = "assets/check.svg"
                return src
            elif self.last_result == 'Fake':
                src = "assets/x.svg"
                return src
            elif self.last_result == 'init':
                src = "assets/init.svg"
                return src



        @app.callback(

            Output('textarea-status-output', 'children'),

            Input('text-pruefen-button', 'n_clicks'),

            State('textarea-status', 'value')

        )
        def update_output(btn1, textarea_value):

            if len(textarea_value) > 0:

                changed_id = [p['prop_id'] for p in callback_context.triggered][0]

                if 'text-pruefen-button' in changed_id:

                    detection = self.detect_fake_news(textarea_value)
                    self.last_result = detection[0]
                    overlap = self.overlap_training_data(textarea_value)


                    return  f'Klassifiziert als: {detection[0]}' \
                            f'\n\n Decision Function: {round( detection[1],2)}' \
                            f'\n\n X_train overlap: {overlap}%'


            return f'\n\n Klassifiziert als:'


        @app.callback(

            Output('textarea-status-output2', 'children'),

            [Input('processing-button', 'n_clicks'),
             Input('model-button', 'n_clicks'),
             Input('input-dropwords', 'value'),],

        )
        def update_output2(btn1, btn2, dropwords_value):

            changed_id = [p['prop_id'] for p in callback_context.triggered][0]

            if 'processing-button' in changed_id:

                dropwords_value = [x.strip().lower() for x in dropwords_value.split(",")]
                self.process(dropwords=dropwords_value)
                return f'\n\n Feature processing abgeschlossen. \nDatei gespeichert als {config.PROCESSED_DATA}.\n' \
                       f'Entfernte Features: {dropwords_value}'

            elif 'model-button' in changed_id:
                self.train_model()
                return f'\n\n Training abgeschlossen.\n\n' \
                       f'Accuracy:\t {self.accuracy_score}\n' \
                       f'Precision:\t {self.precision_score}\n' \
                       f'Recall:\t{self.recall_score}\n' \
                       f'F1-Score:\t {self.f1_score}'


        app.run_server(debug=True)






def main():
    fn = FakeNewsDetector()
    fn.run_dash()


if __name__ == '__main__':
    main()