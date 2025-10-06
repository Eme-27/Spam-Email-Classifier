# test_spam_classifier.py

import unittest
from spam_classifier import train_model, predict_message

class TestSpamClassifier(unittest.TestCase):
    def setUp(self):
        self.model, self.vectorizer, _ = train_model()

    def test_model_trained(self):
        """Check that the model is not None after training"""
        self.assertIsNotNone(self.model)

    def test_vectorizer_trained(self):
        """Check that the vectorizer is not None"""
        self.assertIsNotNone(self.vectorizer)

    def test_predict_spam(self):
        """Check that spam messages are classified as 'spam'."""
        spam_msg = "Congratulations! You've won a free lottery ticket."
        prediction = predict_message(spam_msg, self.model, self.vectorizer)
        self.assertEqual(prediction.lower(), "spam")

    def test_predict_notspam(self):
        """Check that normal messages are classified as 'notspam'."""
        normal_msg = "Hey, are we still meeting for lunch today?"
        prediction = predict_message(normal_msg, self.model, self.vectorizer)
        self.assertEqual(prediction.lower(), "notspam")

if __name__ == "__main__":
    unittest.main()

