from typing import Optional

from openai import OpenAI

from app.config import get_settings
from app.schemas.esg import ESGScores


def generate_esg_insights(scores: ESGScores) -> str:
    settings = get_settings()

    if not settings.openai_api_key:
        return "OpenAI API key is not configured. Cannot generate AI insights."

    client = OpenAI(api_key=settings.openai_api_key)

    prompt = (
        "You are an ESG analyst focusing ONLY on Environmental metrics (energy, carbon, waste, etc.). " +
        "Provide 5 concise insights for operations management. Return 5 bullet points, one per line, no numbering. \n\n"
        f"Company: {scores.company_name}\n"
        f"Period: {scores.period}\n"
        f"E score: {scores.e_score}\n"
        f"S score: {scores.s_score}\n"
        f"G score: {scores.g_score}\n"
        f"Overall ESG score: {scores.overall_score}\n"
    )

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a senior ESG analyst."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
    )

    return completion.choices[0].message.content.strip()
