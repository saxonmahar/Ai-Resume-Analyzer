import os
import sys
import subprocess
import streamlit as st

# ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# auto-train if models don't exist
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
RUN_SCRIPT = os.path.join(BASE_DIR, "run.py")

if not os.path.exists(MODEL_PATH):
    with st.spinner("Setting up models for first time..."):
        subprocess.run([sys.executable, RUN_SCRIPT], cwd=BASE_DIR)

from src.nlp.resume_parser import extract_text_from_pdf
from src.utils.model_loader import load_models
from src.recommender.matcher import (
    recommend_jobs,
    get_resume_score,
    skill_gap_analysis
)

from frontend.components.upload import upload_resume
from src.utils.report_generator import generate_report


# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# -------------------------
# CUSTOM STYLE
# -------------------------
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        h1 { color: #1a1a2e; }
        h2, h3 { color: #16213e; }
    </style>
""", unsafe_allow_html=True)


# -------------------------
# HEADER
# -------------------------
st.title("📄 AI Resume Intelligence System")
st.caption("Upload your resume and get AI-powered job matches, skill gap analysis, and scoring.")
st.divider()


# -------------------------
# UPLOAD RESUME
# -------------------------
resume_path = upload_resume()

if resume_path:

    try:
        with st.spinner("Analyzing your resume..."):

            # Extract text
            resume_text = extract_text_from_pdf(resume_path)

            if not resume_text.strip():
                st.error("Could not extract text from resume.")
                st.stop()

            # Load model
            vectorizer, job_matrix, df = load_models()

            # Vectorize
            resume_vector = vectorizer.transform([resume_text])

            # Score
            score = get_resume_score(resume_vector, job_matrix)

            # Recommendations
            results = recommend_jobs(resume_vector, job_matrix, df)

            if results.empty:
                st.error("No job matches found.")
                st.stop()

            # Top job
            top_job = results.iloc[0]

            # Skill gap
            gap = skill_gap_analysis(resume_text, top_job)

        st.success("Analysis completed successfully!")

        st.divider()


        # -------------------------
        # METRICS
        # -------------------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🎯 Resume Score", f"{score}/100")

        with col2:
            st.metric("💼 Jobs Matched", len(results))

        with col3:
            st.metric("🏆 Top Match", top_job["job title"])


        st.divider()


        # -------------------------
        # SKILL GAP
        # -------------------------
        st.subheader("🧠 Skill Gap Analysis")
        st.caption(f"Top role: **{top_job['job title']}**")

        col1, col2 = st.columns(2)

        with col1:
            st.success("Skills You Have")
            if gap["matched_skills"]:
                for s in gap["matched_skills"]:
                    st.write(f"✔ {s}")
            else:
                st.write("No exact matches found")

        with col2:
            st.error("Skills You Need")
            if gap["missing_skills"]:
                for s in gap["missing_skills"]:
                    st.write(f"➖ {s}")
            else:
                st.write("No missing skills found")


        st.divider()


        # -------------------------
        # JOB RESULTS
        # -------------------------
        st.subheader("🔥 Top Job Matches")

        # convert score to percentage
        results["score"] = (results["score"] * 100).round(2)

        # highlight top match
        st.success(f"Top Match: {results.iloc[0]['job title']} ({results.iloc[0]['score']}%)")

        # rename column
        results = results.rename(columns={"score": "match %"})

        st.dataframe(
            results[["job title", "match %"]],
            use_container_width=True,
            hide_index=True
        )

        st.divider()


        # -------------------------
        # RESUME TEXT
        # -------------------------
        with st.expander("📄 View Extracted Resume Text"):
            st.write(resume_text)


        # -------------------------
        # REPORT DOWNLOAD
        # -------------------------
        st.subheader("📥 Download Report")

        file = generate_report(score, results)

        with open(file, "rb") as f:
            st.download_button(
                label="⬇️ Download Report",
                data=f,
                file_name="resume_report.txt"
            )

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")

else:
    st.info("👆 Upload your resume to start AI analysis.")