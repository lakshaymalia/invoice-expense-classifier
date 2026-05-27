import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib
import os

def train_model():
    print("Loading data...")
    df = pd.read_csv("data.csv")
    
    # Create a pipeline: TF-IDF for preprocessing, Logistic Regression for classification
    # Logistic Regression allows us to extract confidence scores easily using predict_proba
    pipeline = make_pipeline(
        TfidfVectorizer(stop_words='english', lowercase=True),
        LogisticRegression(max_iter=1000)
    )
    
    print("Training model...")
    pipeline.fit(df['text'], df['category'])
    
    print("Saving model...")
    joblib.dump(pipeline, "model.joblib")
    print("Model saved to model.joblib!")

if __name__ == "__main__":
    train_model()