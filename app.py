import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = pickle.load(open('vecterizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


import nltk
from nltk.corpus import stopwords
from nltk.sten import PorterStremmer
import string

nltk.dowload('stopwords')

ps = PorterStremmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    text = [word for word in text if word.isalnum()]

    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    text = [ps.sten(word) for word in text]

    return " ".join(text)


st.title("Email Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transform_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transform_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

