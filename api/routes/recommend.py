from fastapi import APIRouter
from api.schemas.request import RecommendRequest
from api.schemas.response import RecommendResponse
from src.models.predict import predict

router = APIRouter()


@router.post("/", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    results = predict(req.resume_text, req.top_n)
    return RecommendResponse(recommendations=results)
