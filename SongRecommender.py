import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

df = pd.read_csv("spotify_millsongdata.csv")
df = df.sample(20000).drop('link',axis=1).reset_index(drop=True)

df['text'] = df['text'].str.lower().replace(r'^\w\s', '').replace(r'\n', '', regex=True)

stemmer = PorterStemmer()

def token(txt):
    token = nltk.word_tokenize(txt)
    a = [stemmer.stem(w) for w in token]
    return " ".join(a)

df['text'] = df['text'].apply(lambda x: token(x))

tfid = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfid.fit_transform(df['text'])
similiar = cosine_similarity(matrix)

def recommender(song_name):
    idx = df[df['song'] == song_name].index[0]
    distance = sorted(list(enumerate(similiar[idx])), reverse=True, key = lambda x:x[1])
    song = []
    for s_id in distance[1:5]:
        song.append(df.iloc[s_id[0]].song)
    return song

pickle.dump(similiar, open("similiarity", "wb"))
pickle.dump(df, open("df", "wb"))
