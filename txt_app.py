import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from collections import Counter
import re


st.title('TOXIC COMMENTS CLASSIFICATION APP')

df_idf_ = st.file_uploader("CHOOSE CSV FILE WITH IDF INFO")
clf_ = st.file_uploader("CHOOSE CLASSIFIER JOBLIB FILE")


@st.cache
def fn_load():    
    df_idf = pd.read_csv(df_idf_)  
    clf = load(clf_) 
    return df_idf, clf



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



def fn_predict(txt, df_idf, model):

    txt = fn_preprocess_text(txt)
    tf =  dict(Counter(txt.split()))
    tf_vec = np.array([tf.get(word, 0) for word in df_idf.words])
    tfidf_vec = (tf_vec * df_idf.idf.values).reshape(1, -1)

    label = model.predict(tfidf_vec)
    probas = model.predict_proba(tfidf_vec).flatten()
    proba = probas[label]

    return f'label = {label}, proba = {proba}'


if df_idf_ is not None and clf_ is not None:

    df_idf, clf = fn_load()
    sentence = st.text_input("TYPE/COPY/PASTE THE SENTENCE TO BE CLASSIFIED:")
    
    if sentence is not None:
        
        st.write(fn_predict(sentence, df_idf, clf))



