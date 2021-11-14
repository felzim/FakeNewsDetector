import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

from src import config


stopwords = stopwords.words('english')
stopwords = set(stopwords)


df_fake = pd.read_csv(config.DATA_FAKE)
df_true = pd.read_csv(config.DATA_TRUE)


def process(df_true_ = df_true, df_fake_ = df_fake, list_of_dropwords=[], test_size = 0.20):
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
    df_['text'] = df_['text'].apply(lambda x: ' '.join([w for w in x.split() if w not in list_of_dropwords]))

    # bearbeitete Daten als CSV speichern
    df_.to_csv(config.PROCESSED_DATA)

    X = df_['text']
    y = df_['category']

    print(f"y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state= 42)

    print(X_train.shape, y_train.shape)

    return X_train, X_test, y_train, y_test