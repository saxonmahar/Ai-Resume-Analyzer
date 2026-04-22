import os


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def file_extension(filename: str) -> str:
    return os.path.splitext(filename)[-1].lower()
