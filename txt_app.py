import streamlit as st
from joblib import load
import re


def fn_load():
    tfidf_transformer_ = load('./tfidf_transformer.joblib')  
    clf = load('./txt_model_v2.py.joblib') 
    return tfidf_transformer_, clf



def fn_preprocess_text(sentence):

    stop_words = set('and the was to in me my of at it when were by this\
    with that from there one for is we not so are then day had all'.split())

    text = str(sentence).lower()
    
    text = text.replace("won't", "will not").replace("cannot", "can not")
    text = text.replace("what's", "what is").replace("it's", "it is").replace("i'm", "i am")
    text = text.replace("he's", "he is").replace("she's", "she is")
    text = text.replace("'ll", " will").replace("n't", " not").replace("'re", " are")
    text = text.replace("?", "").replace("'ve", " have").replace("can't", "can not")
    
    text = re.sub('[^a-zA-Z\n]', ' ', text) #------------------- Replace every special char with space
    text = re.sub('\s+', ' ', text).strip() #---------------------- Replace excess whitespaces
    
    text = text.split()
    text = [i.lower() for i in text if i.lower() not in stop_words]
    text = ' '.join(text)
    return text



def fn_predict(sentence, tfidf_transformer, clf):

    sentence = fn_preprocess_text(sentence) 
    tfidf_vec = tfidf_transformer.transform([sentence])
    label = clf.predict(tfidf_vec)
    probas = clf.predict_proba(tfidf_vec).flatten()
    proba = probas[label]

    return f'label = {label}, proba = {proba}'



st.title('TOXIC COMMENTS CLASSIFICATION APP')

tfidf_transformer, clf = fn_load()
sentence = st.text_input("TYPE/COPY/PASTE THE SENTENCE TO BE CLASSIFIED:")

if sentence is not None:
    
    y_pred = fn_predict(sentence, tfidf_transformer, clf)          
    st.write(y_pred)



