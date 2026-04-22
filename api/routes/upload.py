from fastapi import APIRouter, UploadFile, File
from src.nlp.resume_parser import parse_resume
import tempfile, os

router = APIRouter()


@router.post("/")
async def upload_resume(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    text = parse_resume(tmp_path)
    os.unlink(tmp_path)
    return {"filename": file.filename, "text_preview": text[:300]}
