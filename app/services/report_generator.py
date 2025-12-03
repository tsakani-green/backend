import io
from datetime import datetime
from fpdf import FPDF
from app.schemas.esg import ESGScores


def generate_esg_report_pdf(scores: ESGScores) -> bytes:
    """
    Generate a simple ESG PDF report from ESGScores.
    Returns PDF as raw bytes.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "ESG Report", ln=True, align="C")

    # Meta
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Company: {scores.company_name}", ln=True)
    pdf.cell(0, 8, f"Period: {scores.period}", ln=True)
    pdf.cell(0, 8, f"Generated: {datetime.utcnow().isoformat()} UTC", ln=True)

    # Scores
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Scores", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"E (Environmental): {scores.e_score}", ln=True)
    pdf.cell(0, 8, f"S (Social): {scores.s_score}", ln=True)
    pdf.cell(0, 8, f"G (Governance): {scores.g_score}", ln=True)
    pdf.cell(0, 8, f"Overall ESG Score: {scores.overall_score}", ln=True)

    # Methodology
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Methodology", ln=True)

    pdf.set_font("Arial", "", 11)
    if scores.methodology:
        for key, desc in scores.methodology.items():
            pdf.multi_cell(0, 6, f"- {key.capitalize()}: {desc}")
    else:
        pdf.multi_cell(0, 6, "Methodology details not available.")

    # Return bytes
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return pdf_bytes
