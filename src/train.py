import pandas as pd
import time

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Classifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


# TFIDF Vectorizer initialisieren
tfvect = TfidfVectorizer(stop_words='english', max_df=0.90)

# PassiveAggressive Classifier initialisieren
classifier = PassiveAggressiveClassifier()

def train_model(df):

    X = df['text']
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # TFIDF Vectorizer verwenden
    #tfvect = TfidfVectorizer(stop_words='english', max_df=0.90)

    # Train und Test mit fit_transform in eine Matrix konvertieren / vektorisieren
    tfid_x_train = tfvect.fit_transform(X_train.astype('U').values)
    tfid_x_test = tfvect.transform(X_test)

    classifier.fit(tfid_x_train, y_train)


def check_fake_news(news):
    ''' Diese Funktion gibt für einen beliebigen String eine eindeutige Klassifizierung, ob True oder Fake
    Dafür nutzt sie den zuvor trainierten Classifier.
    '''


    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = classifier.predict(vectorized_input_data)

    if prediction == 1:
         return "True"
    elif prediction == 0:
        return "Fake"
    else:
        return ""