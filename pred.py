import joblib
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and TF-IDF vectorizer
model = joblib.load('models/spam_classifier_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

def predict_spam(texts):
    # Transform the input texts using the TF-IDF vectorizer
    X = tfidf.transform(texts)
    
    # Predict using the model
    predictions = model.predict(X)
    prediction_probabilities = model.predict_proba(X)
    
    # Convert numeric predictions back to labels (0 = Ham, 1 = Spam)
    predicted_labels = ['ham' if label == 0 else 'spam' for label in predictions]
    
    return predicted_labels, prediction_probabilities

if __name__ == '__main__':
    # Example: Reading from command line arguments
    if len(sys.argv) < 2:
        print("Please provide the text for prediction.")
        sys.exit(1)

    # Get input text from command-line args
    input_text = sys.argv[1:]

    # Make prediction
    predicted_labels, prediction_probs = predict_spam(input_text)

    # Output predictions
    for text, label, prob in zip(input_text, predicted_labels, prediction_probs):
        print(f"Text: {text}")
        print(f"Prediction: {label}")
        print(f"Spam Probability: {prob[1]:.4f}")