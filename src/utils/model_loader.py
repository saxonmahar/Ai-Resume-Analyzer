import pickle
import os


def load_models():
    # check models exist
    if not os.path.exists("models/tfidf_vectorizer.pkl"):
        raise FileNotFoundError("Model not found. Run run.py first to train.")

    vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    job_matrix = pickle.load(open("models/job_matrix.pkl", "rb"))
    df         = pickle.load(open("models/jobs_df.pkl", "rb"))

    print("Models loaded successfully")

    return vectorizer, job_matrix, df
