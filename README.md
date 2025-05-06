# spam-api
Spam Ham Detection 
# Spam Detection API (FastAPI + MLflow + Docker)

This project is a deployable machine learning API for detecting spam messages using a trained classifier. It features:

-  FastAPI for serving predictions
-  MLflow for experiment tracking
-  Docker for containerized deployment
-  Deployable on Render or other cloud platforms

---

## Features

- REST API endpoint to predict if a message is "spam" or "ham"
- Tracks predictions and metrics with MLflow
- Containerized for easy deployment
- Ready for deployment to Render

## Model Info

The model is trained using:

- **TF-IDF Vectorizer** for text preprocessing
- **Logistic Regression / other classifier**
- Saved using `joblib`
