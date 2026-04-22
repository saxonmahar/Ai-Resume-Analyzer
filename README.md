# Resume AI Project

AI-powered resume-to-job matching system using NLP and machine learning.

## Setup

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python -m src.models.train
```

## Run API

```bash
uvicorn api.main:app --reload
```

## Run Frontend

```bash
streamlit run frontend/app.py
```

## Test

```bash
pytest tests/
```
