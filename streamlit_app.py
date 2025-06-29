import json
import random
import nltk
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Load intents
with open('intents.json') as file:
    data = json.load(file)

# Prepare training data
X = []
y = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        X.append(pattern)
        y.append(intent['tag'])

vectorizer = CountVectorizer(tokenizer=word_tokenize, lowercase=True)
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# Define response function
def get_response(user_input):
    inp_vec = vectorizer.transform([user_input])
    pred = model.predict(inp_vec)[0]
    for intent in data['intents']:
        if intent['tag'] == pred:
            return random.choice(intent['responses'])

# Streamlit UI
st.set_page_config(page_title="AI Assistant", layout="centered")
st.title("🎓 AI-Powered Educational Assistant")
st.markdown("Ask me anything about **AI, ML, or Python**:")

user_input = st.text_input("💬 Type your question here:")

if user_input:
    response = get_response(user_input)
    st.success("🤖 " + response)
else:
    st.info("ℹ️ Waiting for your question...")
