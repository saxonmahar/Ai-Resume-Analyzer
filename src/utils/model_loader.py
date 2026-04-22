import pickle
import os

# src/utils/ -> src/ -> project root (2 levels up)
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_models():
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Model not found at {vectorizer_path}. Run run.py first to train.")

    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    job_matrix = pickle.load(open(os.path.join(MODELS_DIR, "job_matrix.pkl"), "rb"))
    df         = pickle.load(open(os.path.join(MODELS_DIR, "jobs_df.pkl"), "rb"))

    print("Models loaded from:", MODELS_DIR)

    return vectorizer, job_matrix, df
