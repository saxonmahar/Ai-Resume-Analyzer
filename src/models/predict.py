import pickle
from src.config import RECOMMENDER_MODEL
from src.recommender.matcher import match


def load_model():
    with open(RECOMMENDER_MODEL, "rb") as f:
        return pickle.load(f)


def predict(resume_text: str, top_n: int = 5):
    model = load_model()
    return match(resume_text, model, top_n)
