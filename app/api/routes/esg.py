from fastapi import APIRouter
from app.schemas.esg import ESGInput, ESGScores
from app.services.esg_calculator import calculate_esg_scores

router = APIRouter()


@router.post("/calculate", response_model=ESGScores)
def calculate_esg(payload: ESGInput):
    """
    Calculate ESG scores for a single company & period.
    """
    return calculate_esg_scores(payload)
