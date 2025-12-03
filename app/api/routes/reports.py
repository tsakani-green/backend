from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.esg import ESGScores
from app.services.report_generator import generate_esg_report_pdf
import io

router = APIRouter()


@router.post("/esg/pdf")
def create_esg_pdf_report(scores: ESGScores):
    """
    Generate a PDF ESG report from scores.
    """
    pdf_bytes = generate_esg_report_pdf(scores)
    buffer = io.BytesIO(pdf_bytes)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=esg_report.pdf"},
    )
