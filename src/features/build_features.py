from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def build_tfidf_model(text_data):
    """
    Convert text into ML features using TF-IDF
    """

    vectorizer = TfidfVectorizer()

    # Convert text → vectors
    tfidf_matrix = vectorizer.fit_transform(text_data)

    print("TF-IDF shape:", tfidf_matrix.shape)

    return vectorizer, tfidf_matrix


def save_model(vectorizer, path):
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)