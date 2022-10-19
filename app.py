import pickle
import gradio as gr
from a_01_text_preprocessing import utils_preprocess_text

def classify_newtext(
    filename_model="lr_model.sav", vectorizer="tfidf.pickle"
):
    with open(filename_model, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer
    
model_0, vectorizer_0 = classify_newtext()
def predict(input_text, model = model_0, vectorizer = vectorizer_0):    
    text = utils_preprocess_text(input_text)
    vector_test = vectorizer.transform([text])
    predicted = model.predict(vector_test)
    if predicted[0] == 1:
        return "suicide attempt"
    else:
        return "no obvious suicide attempt"
gr.Interface(fn=predict, inputs="text", outputs="text").launch()