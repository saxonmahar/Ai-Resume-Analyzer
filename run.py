import pandas as pd
import pickle
import os

from src.data.preprocessing import preprocess_jobs
from src.features.build_features import build_tfidf_model
from src.recommender.matcher import recommend_jobs

# resolve paths relative to this file
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "raw", "job_roles.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Preprocess
df = preprocess_jobs(df)

# TF-IDF training
texts = df["combined_text"].tolist()
vectorizer, job_matrix = build_tfidf_model(texts)

# Save models
pickle.dump(vectorizer, open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb"))
pickle.dump(job_matrix, open(os.path.join(MODELS_DIR, "job_matrix.pkl"), "wb"))
pickle.dump(df,         open(os.path.join(MODELS_DIR, "jobs_df.pkl"), "wb"))

print("✅ Model saved successfully")
