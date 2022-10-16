from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import pickle
import pandas as pd
import re
import nltk

"""This code aims to transform texts into tfidf vectors and save them for future usage."""


def load(filepath="processed_trainData.csv"):
    df = pd.read_csv(filepath)
    df = df.dropna()
    return df


def corpus_label_from_df(df):
    corpus = df["text_clean"].tolist()
    label = df["y"].tolist()
    return corpus, label


def tfidf_transform(corpus, label):
    tfidf_vectorizer = TfidfVectorizer().fit(corpus)
    text_representation = tfidf_vectorizer.transform(corpus)
    pickle.dump(tfidf_vectorizer, open("tfidf.pickle", "wb"))
    pickle.dump(text_representation, open("text_representation.pickle", "wb"))
    pickle.dump(corpus, open("corpus.pickle", "wb"))
    pickle.dump(label, open("label.pickle", "wb"))


def main():
    df = load()
    corpus, label = corpus_label_from_df(df)
    tfidf_transform(corpus, label)


if __name__ == "__main__":
    main()
