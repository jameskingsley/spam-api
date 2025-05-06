# Use official Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy app code
COPY ./app /app
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install MLflow
RUN pip install mlflow

# Expose FastAPI and MLflow UI ports
EXPOSE 8000 5000

# Start FastAPI app and MLflow UI (in the background)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & mlflow ui --host 0.0.0.0 --port 5000"]
