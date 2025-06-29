import json
import random
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')

# Load intents
with open('intents.json') as file:
    data = json.load(file)

X = []
y = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        X.append(pattern)
        y.append(intent['tag'])

# Vectorize using NLTK tokenizer
from nltk.tokenize import word_tokenize
vectorizer = CountVectorizer(tokenizer=word_tokenize, lowercase=True)
X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Chat loop
def chat():
    print("🤖 AI Assistant: Hello! Ask me anything about AI, ML, or Python. Type 'exit' to leave.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("🤖 AI Assistant: Bye! Keep learning.")
            break
        inp_vec = vectorizer.transform([user_input])
        prediction = model.predict(inp_vec)[0]
        for intent in data['intents']:
            if intent['tag'] == prediction:
                print("🤖 AI Assistant:", random.choice(intent['responses']))

chat()
