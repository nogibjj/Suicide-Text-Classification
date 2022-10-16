import nltk
import numpy as np
import pandas as pd
import re

"""This code aims to perform text preprocessing and save processed texts as a new file"""


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    """Text processing: remove stopwords, stem or lemma"""
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r"[^\w\s]", "", str(text).lower().strip())

    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list
    text = " ".join(lst_text)
    return text


def main(
    file_path="Suicide_Detection.csv",
    lst_stopwords=nltk.corpus.stopwords.words("english"),
):
    df = pd.read_csv(file_path, index_col=False)
    df = df.iloc[:, 1:]
    # class transformation to 0 and 1
    df["y"] = df["class"].map({"suicide": "1", "non-suicide": "0"})
    df["text_clean"] = df["text"].apply(
        lambda x: utils_preprocess_text(
            x, flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords
        )
    )
    df.to_csv("processed_trainData.csv", index=False)


if __name__ == "__main__":
    main()
