import pandas as pd
from src.config import JOB_ROLES_CSV, CLEANED_JOBS_CSV


def load_raw_jobs() -> pd.DataFrame:
    return pd.read_csv(JOB_ROLES_CSV)


def load_cleaned_jobs() -> pd.DataFrame:
    return pd.read_csv(CLEANED_JOBS_CSV)
