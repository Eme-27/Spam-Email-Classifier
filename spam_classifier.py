import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data/emails.csv")

# Split into features and labels
X = data["text"]
y = data["label"]

# Convert text into numeric vectors
vectorizer = CountVectorizer()
X_vectors = vectorizer.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_vectors, y, test_size=0.3, random_state=42
)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully with accuracy: {accuracy * 100:.2f}%")

# Interactive testing
while True:
    user_input = input("\nEnter an email message (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    user_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_vector)[0]
    print(f"📧 This message is likely: {prediction}")
