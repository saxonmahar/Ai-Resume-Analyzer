import pandas as pd
import pickle

from src.data.preprocessing import preprocess_jobs
from src.features.build_features import build_tfidf_model
from src.recommender.matcher import recommend_jobs


# Load dataset
df = pd.read_csv("data/raw/job_roles.csv")

# Preprocess
df = preprocess_jobs(df)

# TF-IDF training
texts = df["combined_text"].tolist()
vectorizer, job_matrix = build_tfidf_model(texts)

# SAVE MODEL (IMPORTANT)
pickle.dump(vectorizer, open("models/tfidf_vectorizer.pkl", "wb"))
pickle.dump(job_matrix, open("models/job_matrix.pkl", "wb"))
pickle.dump(df, open("models/jobs_df.pkl", "wb"))

print("✅ Model saved successfully")