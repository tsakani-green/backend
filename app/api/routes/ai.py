from fastapi import APIRouter
from app.schemas.esg import ESGScores
from app.services.ai_insights import generate_esg_insights

router = APIRouter()


@router.post("/esg/insights")
def esg_ai_insights(scores: ESGScores):
    """
    Generate AI-based narrative ESG insights for the given scores.
    """
    insights = generate_esg_insights(scores)
    return {"insights": insights}
