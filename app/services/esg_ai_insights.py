# app/services/esg_ai_insights.py

import json
import os
from typing import Dict

from openai import OpenAI

from app.schemas.esg import ESGScores


# ---------- OpenAI client initialisation ----------

API_KEY = os.getenv("OPENAI_API_KEY")

if API_KEY:
    client = OpenAI(api_key=API_KEY)
else:
    client = None
    print(
        "[esg_ai_insights] WARNING: OPENAI_API_KEY is not set. "
        "Falling back to rule-based insights."
    )


# ---------- Fallback (no OpenAI / API error) ----------

def _fallback_insights(scores: ESGScores) -> Dict[str, object]:
    """
    Simple rule-based insights used when OpenAI is not configured
    or when the API call fails, so the dashboard still has content.
    """

    e = scores.e_score
    s = scores.s_score
    g = scores.g_score
    overall = scores.overall_score

    # Overall narrative
    if overall >= 80:
        overall_txt = (
            f"{scores.company_name} shows strong ESG performance in the {scores.period} "
            f"period, with an overall score of {overall}. The organisation is generally "
            "well positioned on sustainability fundamentals and can now focus on "
            "targeted improvements and scale-up of best practices."
        )
    elif overall >= 60:
        overall_txt = (
            f"{scores.company_name} demonstrates moderate ESG performance in {scores.period}, "
            f"with an overall score of {overall}. Core structures are in place, but there "
            "are clear opportunities to improve environmental efficiency and strengthen "
            "social and governance practices."
        )
    else:
        overall_txt = (
            f"{scores.company_name} has a relatively weak ESG profile in {scores.period}, "
            f"with an overall score of {overall}. Material improvements are needed across "
            "environmental, social and governance areas to align with investor and "
            "regulatory expectations."
        )

    # Environmental bullets
    env = []
    if e >= 80:
        env.append(
            "Environmental performance is strong, with relatively efficient use of energy and resources."
        )
        env.append(
            "Consider targeting deeper decarbonisation through renewables and low-carbon technologies."
        )
    elif e >= 60:
        env.append(
            "Environmental score is moderate, indicating room to improve energy efficiency and emissions management."
        )
        env.append(
            "Prioritise a clear decarbonisation roadmap with short-term, measurable milestones."
        )
    else:
        env.append(
            "Environmental performance is weak relative to leading peers; emissions and resource use appear high."
        )
        env.append(
            "An accelerated programme on energy efficiency, cleaner fuels and process optimisation is recommended."
        )

    # Social bullets
    soc = []
    if s >= 80:
        soc.append(
            "Social performance is strong, indicating positive employee, community and stakeholder outcomes."
        )
        soc.append(
            "Maintain current programmes and expand high-impact initiatives such as local supplier development."
        )
    elif s >= 60:
        soc.append(
            "Social score suggests a reasonable baseline, but more structured community and employee programmes are needed."
        )
        soc.append(
            "Enhance tracking of workforce wellbeing, diversity and local economic impact."
        )
    else:
        soc.append(
            "Social performance appears weak, raising potential risks around workforce stability and social licence to operate."
        )
        soc.append(
            "Develop targeted interventions on safety, labour relations and local community engagement."
        )

    # Governance bullets
    gov = []
    if g >= 80:
        gov.append(
            "Governance structures appear robust, with strong oversight and control frameworks."
        )
        gov.append(
            "Leverage this governance strength to drive the ESG and climate transition agenda more aggressively."
        )
    elif g >= 60:
        gov.append(
            "Governance practices are adequate but show room for improvement in board oversight and transparency."
        )
        gov.append(
            "Clarify ESG responsibilities at board and executive level, and strengthen disclosure practices."
        )
    else:
        gov.append(
            "Governance performance is weak and may expose the organisation to compliance and reputation risks."
        )
        gov.append(
            "Prioritise board independence, risk management and control improvements to stabilise the ESG foundation."
        )

    return {
        "overall": overall_txt,
        "environmental": env,
        "social": soc,
        "governance": gov,
    }


# ---------- OpenAI-based ESG insights ----------

def generate_esg_insights(scores: ESGScores) -> Dict[str, object]:
    """
    Call an LLM to generate structured ESG insights for the dashboard.
    If OpenAI is not configured or the call fails, fall back to rule-based insights.

    Returns a dict like:
    {
        "overall": "High-level narrative...",
        "environmental": ["bullet 1", "bullet 2", ...],
        "social": ["bullet 1", ...],
        "governance": ["bullet 1", ...]
    }
    """

    # If no client (no API key), use fallback
    if client is None:
        return _fallback_insights(scores)

    prompt = f"""
You are an ESG analyst. You are given ESG scores for a company.
Write concise, practical insights for an Africa-focused ESG dashboard.

Company: {scores.company_name}
Period: {scores.period}

Scores (0-100):
- Environmental (E): {scores.e_score}
- Social (S): {scores.s_score}
- Governance (G): {scores.g_score}
- Overall: {scores.overall_score}

Methodology (for context):
{json.dumps(scores.methodology, indent=2)}

Produce:
1. A short OVERALL narrative (3–5 sentences) for executives.
2. 3–5 Environmental bullets (practical, specific).
3. 3–5 Social bullets.
4. 3–5 Governance bullets.

Keep it region-agnostic but realistic for African corporates.
    """.strip()

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert ESG analyst. Always respond in valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "EsgInsights",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "overall": {
                                "type": "string",
                                "description": "Short narrative (3–5 sentences) summarising ESG performance.",
                            },
                            "environmental": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Environmental insights as bullet points.",
                            },
                            "social": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Social insights as bullet points.",
                            },
                            "governance": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Governance insights as bullet points.",
                            },
                        },
                        "required": [
                            "overall",
                            "environmental",
                            "social",
                            "governance",
                        ],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        )

        raw = completion.choices[0].message.content
        data = json.loads(raw)

        return {
            "overall": data.get("overall", ""),
            "environmental": data.get("environmental") or [],
            "social": data.get("social") or [],
            "governance": data.get("governance") or [],
        }

    except Exception as e:
        # If anything goes wrong (bad key, network, model error),
        # log it and fall back so the dashboard never breaks.
        print(f"[esg_ai_insights] ERROR calling OpenAI: {e}")
        return _fallback_insights(scores)
