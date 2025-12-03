from pydantic import BaseModel, Field
from typing import Optional, Dict


class ESGInput(BaseModel):
    company_name: str = Field(..., example="GreenBDG Africa")
    period: str = Field(..., example="2025-Q1")

    carbon_emissions_tons: float = Field(..., ge=0, example=1200.5)
    energy_consumption_mwh: float = Field(..., ge=0, example=4500.0)
    water_use_m3: float = Field(..., ge=0, example=8000.0)
    waste_generated_tons: float = Field(..., ge=0, example=150.0)

    social_score_raw: float = Field(..., ge=0, le=100, example=78.0)
    governance_score_raw: float = Field(..., ge=0, le=100, example=82.0)


class ESGScores(BaseModel):
    company_name: str
    period: str
    e_score: float
    s_score: float
    g_score: float
    overall_score: float
    methodology: Optional[Dict[str, str]] = None
