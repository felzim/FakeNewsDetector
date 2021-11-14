from src import config

import re

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

import dash
from dash import callback_context
from dash.dependencies import Input, Output, State
from dash import html
from dash import dcc

import pandas as pd
from sklearn.model_selection import train_test_split
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

        # TFIDF Vectorizer initialisieren
        self.tfvect = TfidfVectorizer(stop_words='english', max_df=0.90)

        # PassiveAggressive Classifier initialisieren
        self.classifier = PassiveAggressiveClassifier()

        self.process()
        self.train_model()


    def train_model(self):

        # Train und Test mit fit_transform in eine Matrix konvertieren / vektorisieren
        self.tfidf_x_train = self.tfvect.fit_transform(self.X_train.astype('U').values)

        # Classifier trainieren
        self.classifier.fit(self.tfidf_x_train, self.y_train)


    def detect_fake_news(self, news):
        ''' Diese Funktion gibt für einen beliebigen String eine eindeutige Klassifizierung, ob True oder Fake
        Dafür nutzt sie den zuvor trainierten Classifier.
        '''

        input_data = [news]

        # eingegebenen Text mit dem bereits gefitteten Modell in numerisches Format transformieren
        vectorized_input_data = self.tfvect.transform(input_data)
        prediction = self.classifier2.predict(vectorized_input_data)[0]

        if prediction == 0:
            return ["Fake", self.classifier2.decision_function(vectorized_input_data)[0]]
        if prediction == 1:
            return ["True", self.classifier2.decision_function(vectorized_input_data)[0]]

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


    def process(self, df_true_= df_true, df_fake_=df_fake, dropwords=[], test_size=0.20):
        ''' Funktion, die aus zwei Dataframes einen neuen zusammensetzt und diesen bereinigt
            Es kann optional eine Liste von Wörtern mit übergeben werden, die dann aus dem neuen DF herausgefiltert werden
        '''

        # vor dem Zusammensetzen das target festschreiben
        df_true_['category'] = 1
        df_fake_['category'] = 0

        # neuen Dataframe mit true and fake
        df_ = pd.concat([df_true_, df_fake_])

        # neuen Index erstellen und alten Index droppen
        df_ = df_.reset_index(drop=False)
        df_ = df_.drop(columns='index')

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

        # wenn eine Dropliste mitgegeben wurde, diese aus dem neuen DF entfernen
        #df_['text'] = df_['text'].apply(lambda x: ' '.join([w for w in x.split() if w not in list_of_dropwords]))

        for word in dropwords:
            df_['text'] = df_['text'].str.replace(word, '')

        # bearbeitete Daten als CSV speichern
        df_.to_csv(config.PROCESSED_DATA)

        X = df_['text']
        y = df_['category']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)


    # Web-Interface auf 127.0.0.1:8050
    def run_dash(self):

        app = dash.Dash(__name__)

        app.layout = html.Div([


            html.Link(
                rel='stylesheet',
                href='assets/custom_loader.css'
            ),

            html.H2('Hi, I\'m a very simple Fake News Detector', style={'textAlign': 'center', 'margin-bottom': '2em'}),

            html.Div([

                html.Button('Text prüfen', id='text-pruefen-button', n_clicks=0),
                html.Button('Modell neu berechnen', id='model-button', style={'float': 'right'}, n_clicks=0),
                html.Br(),
                dcc.Textarea(
                    id='textarea-status',
                    value='Bitte einen englischsprachigen Text eingeben.',
                    style={'width': '100%','height': 200, 'margin-top': '1em', 'padding': '4px'},
                ),

                html.Div(id='textarea-status-output', style={'whiteSpace': 'pre-line',
                                                             'font-size': '0.9em',
                                                             'height': 150}),

                html.H4("Optionen: "),
                html.Span("Features aus Datensatz entfernen: "),

                html.Br(),
                dcc.Input(
                    id='input-dropwords',
                    placeholder='Wörter durch Komma getrennt eingeben...',
                    type='text',
                    value='',
                    style={'width': '50%', 'margin-top': '1em'}
                ),
                html.Button('Feature processing ausführen', id='processing-button', n_clicks=0),

            ],
            style={'width': '40%'}),
        ])

        @app.callback(

            Output('textarea-status-output', 'children'),

            [Input('text-pruefen-button', 'n_clicks'),
             Input('processing-button', 'n_clicks'),
             Input('model-button', 'n_clicks'),
             Input('input-dropwords', 'value')],

            State('textarea-status', 'value')

        )
        def update_output(btn1, btn2, btn3, dropwords_value, textarea_value):

            if len(textarea_value) > 0:

                changed_id = [p['prop_id'] for p in callback_context.triggered][0]

                if 'text-pruefen-button' in changed_id:

                    detection = self.detect_fake_news(textarea_value)
                    overlap = self.overlap_training_data(textarea_value)

                    return  f'\n\n Klassifiziert als: {detection[0]}' \
                            f'\n\n Decision Function: {round( detection[1],2)}' \
                            f'\n\n Overlap: {overlap}% der eingegebenen Wörter sind in den Trainingsdaten enthalten.'

                elif 'processing-button' in changed_id:
                    dropwords_value = [x.strip().lower() for x in dropwords_value.split(",")]
                    self.process(dropwords=dropwords_value)
                    return f'\n\n Feature processing abgeschlossen. \nDatei gespeichert als {config.PROCESSED_DATA}.\n' \
                           f'Entfernte Features: {dropwords_value}'

                elif 'model-button' in changed_id:
                    self.train_model()
                    return f'\n\n Training abgeschlossen.'

        app.run_server(debug=True)



def main():
    fn = FakeNewsDetector()
    fn.run_dash()


if __name__ == '__main__':
    main()