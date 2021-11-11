import pandas as pd
import time
import re

# Sklearn und Classifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


# TFIDF Vectorizer initialisieren
tfvect = TfidfVectorizer(stop_words='english', max_df=0.90)

# PassiveAggressive Classifier initialisieren
classifier = PassiveAggressiveClassifier()


def train_model(X_train, y_train):

    # Train und Test mit fit_transform in eine Matrix konvertieren / vektorisieren
    tfidf_x_train = tfvect.fit_transform(X_train.astype('U').values)

    # Classifier trainieren
    classifier.fit(tfidf_x_train, y_train)


def detect_fake_news(news):
    ''' Diese Funktion gibt für einen beliebigen String eine eindeutige Klassifizierung, ob True oder Fake
    Dafür nutzt sie den zuvor trainierten Classifier.
    '''

    input_data = [news]

    # eingegebenen Text mit dem bereits gefitteten Modell in numerisches Format transformieren
    vectorized_input_data = tfvect.transform(input_data)
    prediction = classifier.predict(vectorized_input_data)

    if prediction == 1:
         return "True"
    elif prediction == 0:
        return "Fake"
    else:
        return ""


def overlap_training_data(news, X_train):
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
        res = set(X_train.str.contains(word.lower(), regex=False))
        if True in res:
            count_enthalten += 1
        else:
            count_not_enthalten += 1

    return round((count_enthalten / len(list_of_words) * 100), 2)