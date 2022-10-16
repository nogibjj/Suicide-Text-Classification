from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

"""Train a Logistic Regression model with all texts in the corpus and save the model."""


def load_tfidf_label():
    with open("tfidf.pickle", "rb") as f:
        tfidf_vec = pickle.load(f)
    with open("text_representation.pickle", "rb") as f:
        text_representation = pickle.load(f)
    with open("label.pickle", "rb") as f:
        label = pickle.load(f)
    with open("corpus.pickle", "rb") as f:
        corpus = pickle.load(f)
    return tfidf_vec, text_representation, corpus, label


def train_save_model(tfidf_vec, text_representation, label):
    lr_model = LogisticRegression(max_iter=500)
    lr_model.fit(text_representation, np.array(label))
    filename = "lr_model.sav"
    pickle.dump(lr_model, open(filename, "wb"))


def main():
    tfidf_vec, text_representation, corpus, label = load_tfidf_label()
    train_save_model(tfidf_vec, text_representation, label)


if __name__ == "__main__":
    main()
