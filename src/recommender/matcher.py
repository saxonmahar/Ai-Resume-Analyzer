from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# -------------------------
# RECOMMEND JOBS
# -------------------------
def recommend_jobs(resume_vector, job_matrix, df, top_k=5, threshold=0.1):

    scores = cosine_similarity(resume_vector, job_matrix).flatten()

    df = df.copy()
    df["score"] = scores

    # count jobs above threshold as "matched"
    matched_count = int((df["score"] >= threshold).sum())

    results = df.sort_values(by="score", ascending=False).head(top_k)

    return results, matched_count


# -------------------------
# RESUME SCORE (0–100)
# -------------------------
def get_resume_score(resume_vector, job_matrix):

    scores = cosine_similarity(resume_vector, job_matrix).flatten()

    # use mean + max for stability (REAL WORLD FIX)
    score = (np.max(scores) * 0.7) + (np.mean(scores) * 0.3)

    return round(float(score * 100), 2)


# -------------------------
# SKILL GAP ANALYSIS (IMPROVED)
# -------------------------
def skill_gap_analysis(resume_text, top_job_row):

    # clean resume words better
    resume_words = set(
        word.strip().lower()
        for word in resume_text.split()
        if len(word) > 2
    )

    # clean job skills better
    job_skills = str(top_job_row["required skills"]).lower()
    job_skills = set(
        skill.strip()
        for skill in job_skills.split("|")
        if skill.strip()
    )

    matched = resume_words.intersection(job_skills)
    missing = job_skills - resume_words

    return {
        "matched_skills": sorted(list(matched)),
        "missing_skills": sorted(list(missing))
    }