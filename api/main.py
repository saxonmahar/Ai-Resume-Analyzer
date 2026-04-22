from fastapi import FastAPI, UploadFile, File
import shutil

from src.nlp.resume_parser import extract_text_from_pdf
from src.utils.model_loader import load_models
from src.recommender.matcher import recommend_jobs

app = FastAPI(title="Resume AI API")

# Load model ONCE (important for performance)
vectorizer, job_matrix, df = load_models()


@app.get("/")
def home():
    return {"message": "Resume AI API is running 🚀"}


# -------------------------
# UPLOAD RESUME API
# -------------------------
@app.post("/upload_resume/")
def upload_resume(file: UploadFile = File(...)):

    file_path = "temp.pdf"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    resume_text = extract_text_from_pdf(file_path)

    return {"resume_text": resume_text}


# -------------------------
# RECOMMEND JOBS API
# -------------------------
@app.post("/recommend_jobs/")
def recommend(resume_text: str):

    resume_vector = vectorizer.transform([resume_text])

    results = recommend_jobs(resume_vector, job_matrix, df)

    return results[["job title", "score"]].to_dict(orient="records")