from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

JOB_ROLES_CSV = DATA_RAW / "job_roles.csv"
CLEANED_JOBS_CSV = DATA_PROCESSED / "cleaned_jobs.csv"
FEATURES_PKL = DATA_PROCESSED / "features.pkl"

TFIDF_MODEL = MODELS_DIR / "tfidf_vectorizer.pkl"
RECOMMENDER_MODEL = MODELS_DIR / "recommender_model.pkl"
