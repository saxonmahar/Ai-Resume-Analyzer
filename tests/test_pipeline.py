import pandas as pd
from src.data.preprocessing import preprocess_jobs
from src.nlp.skill_extraction import extract_skills


def test_preprocess_jobs():
    # create a small test dataframe
    data = {
        "Job Title": ["Data Scientist", "Web Developer"],
        "Category": ["Technology", "Technology"],
        "Required Skills": ["Python|SQL|ML", "JavaScript|HTML|CSS"],
        "Education Requirement": ["Bachelor's", "Bachelor's"],
        "Experience Years": [2, 1],
        "Salary Range": ["80-120K", "60-100K"]
    }
    df = pd.DataFrame(data)
    result = preprocess_jobs(df)

    assert result is not None
    assert "combined_text" in result.columns
    assert len(result) == 2


def test_extract_skills():
    skills = extract_skills("I know Python and SQL")
    assert "python" in skills
    assert "sql" in skills
