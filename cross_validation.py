from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import metrics
import re
import nltk

"""This code perform cross validation on the dataset using Logistic Regression model"""
"""This validation process take around 5-10 minutes"""


def cross_validation(corpus, label):
    predicted = []
    expected = []
    for train_index, test_index in KFold(n_splits=10, shuffle=True).split(corpus):
        x_train = np.array(corpus)[train_index]
        y_train = np.array(label)[train_index]
        x_test = np.array(corpus)[test_index]
        y_test = np.array(label)[test_index]
        tfidf_bow = TfidfVectorizer()
        vector_training = tfidf_bow.fit_transform(x_train)
        vector_test = tfidf_bow.transform(x_test)
        lr_model = LogisticRegression(max_iter=500)
        lr_model.fit(vector_training, y_train)
        expected.extend(y_test)
        predicted.extend(lr_model.predict(vector_test))

    print(
        "Macro-average: {0}".format(
            metrics.f1_score(expected, predicted, average="macro")
        )
    )
    print(
        "Micro-average: {0}".format(
            metrics.f1_score(expected, predicted, average="micro")
        )
    )
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))


def main():
    with open("label.pickle", "rb") as f:
        label = pickle.load(f)
    with open("corpus.pickle", "rb") as f:
        corpus = pickle.load(f)
    cross_validation(corpus, label)


if __name__ == "__main__":
    main()
