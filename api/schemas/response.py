from pydantic import BaseModel


class JobMatch(BaseModel):
    job_title: str
    score: float


class RecommendResponse(BaseModel):
    recommendations: list[JobMatch]
