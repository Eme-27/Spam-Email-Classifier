# spam_classifier.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_model():
    """
    Loads data, trains a spam classifier, and returns the model and vectorizer.
    """
    # Load CSV
    data = pd.read_csv("data/emails.csv")

    # Features and labels
    X_text = data['text']
    y = data['label']

    # Vectorize text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X_text)

    # Train model
    model = MultinomialNB()
    model.fit(X, y)

    print(f"✅ Model trained successfully with accuracy: {model.score(X, y) * 100:.2f}%")
    return model, vectorizer, data

def predict_message(message, model, vectorizer):
    """
    Predicts whether a single message is spam or not.
    """
    X = vectorizer.transform([message])  # Use transform, NOT fit_transform
    prediction = model.predict(X)[0]
    return prediction

if __name__ == "__main__":
    model, vectorizer, _ = train_model()

    while True:
        msg = input("\nEnter an email message (or 'exit' to quit): ")
        if msg.lower() == "exit":
            break
        result = predict_message(msg, model, vectorizer)
        print(f"Prediction: {result}")
