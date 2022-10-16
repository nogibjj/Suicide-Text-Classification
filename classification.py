from text_preprocessing import utils_preprocess_text
import pickle


def classify_newtext(
    input_text, filename_model="lr_model.sav", vectorizer="tfidf.pickle"
):
    with open(filename_model, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer, "rb") as f:
        vectorizer = pickle.load(f)
    text = utils_preprocess_text(input_text)
    vector_test = vectorizer.transform([text])
    predicted = model.predict(vector_test)
    if predicted[0] == 1:
        print("suicide attempt")
    else:
        print("no obvious suicide attempt")


def main():
    input_text = input()
    classify_newtext(
        input_text, filename_model="lr_model.sav", vectorizer="tfidf.pickle"
    )


if __name__ == "__main__":
    main()
