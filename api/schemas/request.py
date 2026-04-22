from pydantic import BaseModel


class RecommendRequest(BaseModel):
    resume_text: str
    top_n: int = 5
