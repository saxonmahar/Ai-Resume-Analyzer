SKILLS_DB = [
    "python", "java", "c++", "sql", "machine learning",
    "deep learning", "nlp", "tensorflow", "pytorch",
    "aws", "docker", "kubernetes", "git", "flask", "fastapi"
]


def extract_skills(text):
    text = text.lower()
    found_skills = []

    for skill in SKILLS_DB:
        if skill in text:
            found_skills.append(skill)

    return list(set(found_skills))