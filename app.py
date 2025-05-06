import joblib
import re
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load the pre-trained model and TF-IDF vectorizer
model = joblib.load('spam_classifier_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Initialize FastAPI app
app = FastAPI()

# Set MLflow experiment
mlflow.set_experiment("spam-detection")

class Message(BaseModel):
    text: str

def sanitize_text(text: str, max_length: int = 100) -> str:
    """Sanitize and truncate input text for logging."""
    # Remove newlines and extra spaces
    sanitized = re.sub(r'\s+', ' ', text.strip())
    # Optionally mask email addresses or URLs
    sanitized = re.sub(r'\S+@\S+\.\S+', '[email]', sanitized)
    sanitized = re.sub(r'https?://\S+', '[url]', sanitized)
    # Truncate to max_length
    return sanitized[:max_length]

@app.post("/predict/")
def predict_spam(message: Message):
    try:
        with mlflow.start_run(nested=True):
            # Log a sanitized version of the input text
            safe_text = sanitize_text(message.text)
            mlflow.log_param("input_text_sanitized", safe_text)

            # Preprocess and transform input text
            text = [message.text]
            X = tfidf.transform(text)

            # Make prediction
            prediction = model.predict(X)[0]
            prediction_prob = model.predict_proba(X)[0][1]

            # Convert to label
            label = "spam" if prediction == 1 else "ham"

            # Log prediction info
            mlflow.log_metric("prediction_probability", prediction_prob)
            mlflow.log_param("predicted_label", label)

            return {
                "message": message.text,
                "prediction": label,
                "spam_probability": prediction_prob
            }

    except Exception as e:
        mlflow.log_param("error", str(e))
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Spam Detection API!"}
