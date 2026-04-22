import pickle
from src.data.loader import load_cleaned_jobs
from src.features.build_features import build_tfidf
from src.config import RECOMMENDER_MODEL


def train():
    df = load_cleaned_jobs()
    corpus = df["cleaned_description"].tolist()
    vectorizer, matrix = build_tfidf(corpus)

    model = {"vectorizer": vectorizer, "matrix": matrix, "jobs": df}
    with open(RECOMMENDER_MODEL, "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved.")


if __name__ == "__main__":
    train()
