import pandas as pd


def preprocess_jobs(df):
    # Step 1: make all column names lowercase
    df.columns = [col.lower() for col in df.columns]

    print("Columns:", df.columns)

    # Step 2: check required columns exist
    if "job title" not in df.columns:
        print("Error: Job Title column missing")
        return None

    if "required skills" not in df.columns:
        print("Error: Required Skills column missing")
        return None

    # Step 3: strip whitespace from key columns
    df["job title"] = df["job title"].str.strip()
    df["required skills"] = df["required skills"].str.strip()
    df["category"] = df["category"].str.strip()

    # Step 4: split pipe-separated skills into space-separated string
    # e.g. "Python|SQL|Excel" → "Python SQL Excel"
    df["required skills"] = df["required skills"].str.replace("|", " ", regex=False)

    # Step 5: create combined text (important for ML later)
    df["combined_text"] = df["job title"] + " " + df["category"] + " " + df["required skills"]

    # Step 6: remove missing values only in critical columns
    df = df.dropna(subset=["job title", "required skills"])

    # Step 7: remove duplicate rows
    df = df.drop_duplicates()

    print("Data cleaned successfully")
    print("Total jobs:", len(df))

    return df
