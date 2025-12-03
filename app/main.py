import os
import io
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import base64
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, WebSocket
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

load_dotenv()

# PDF reader (requires `python -m pip install pypdf`)
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# Image/PDF engine for logo extraction (requires `pip install pymupdf`)
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# ================== CONFIG ==================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL_ESG", "gpt-4o-mini")  # default model

FRONTEND_ORIGINS_ENV = os.getenv("FRONTEND_ORIGINS")
if FRONTEND_ORIGINS_ENV:
    ALLOWED_ORIGINS = [o.strip() for o in FRONTEND_ORIGINS_ENV.split(",")]
else:
    ALLOWED_ORIGINS = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "https://esg-dashboard-cznr.vercel.app",
        "*",
    ]

# Where to store the "last ESG input" snapshot on disk
DATA_DIR = os.getenv("ESG_DATA_DIR", ".")
os.makedirs(DATA_DIR, exist_ok=True)
LAST_ESG_JSON_PATH = os.path.join(DATA_DIR, "last_esg_input.json")

# Where to store the raw uploaded ESG rows (for charts)
LAST_ESG_ROWS_PATH = os.path.join(DATA_DIR, "last_esg_uploaded_rows.json")

# Optional OpenAI client (Chat Completions)
try:
    from openai import OpenAI

    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except ImportError:
    openai_client = None


# ================== MODELS ==================
class ESGInput(BaseModel):
    """
    Core ESG input structure used for analysis and mock data generation.
    """

    company_name: str = Field(..., example="GreenBDG Africa")
    period: str = Field(..., example="2025-Q1")

    carbon_emissions_tons: float = Field(..., ge=0, example=18500.0)
    energy_consumption_mwh: float = Field(..., ge=0, example=1250.0)
    water_use_m3: float = Field(..., ge=0, example=55000.0)
    waste_generated_tons: float = Field(..., ge=0, example=180.0)

    # total fuel usage in litres (populated when Excel has Fuel (L))
    fuel_litres: float = Field(0.0, ge=0, example=50000.0)

    social_score_raw: float = Field(..., ge=0, le=100, example=78.0)
    governance_score_raw: float = Field(..., ge=0, le=100, example=82.0)


class ESGScores(BaseModel):
    company_name: str
    period: str
    e_score: float
    s_score: float
    g_score: float
    overall_score: float

    # Automatically generated when calculating scores
    methodology: Optional[Dict[str, str]] = Field(
        default=None,
        example={
            "environmental": "Based on emission intensity and energy efficiency with fixed weights.",
            "social": "Directly derived from social_score_raw (0–100).",
            "governance": "Directly derived from governance_score_raw (0–100).",
            "overall": "Simple average of E, S, and G scores.",
        },
    )


class ESGInsights(BaseModel):
    overall: str
    environmental: List[str]
    social: List[str]
    governance: List[str]


class AnalyseResponse(BaseModel):
    scores: ESGScores
    insights: ESGInsights


class PlatformOverview(BaseModel):
    countries_supported: int
    esg_reports_generated: int
    compliance_accuracy: float
    ai_support_mode: str


class PillarInsightsResponse(BaseModel):
    metrics: Dict[str, Any]
    insights: List[str]


class SocialInsightsRequest(BaseModel):
    """
    Request body for POST /api/social-insights.
    Allows the frontend to send the current social metrics
    (e.g. from SimulationContext) for live AI analysis.
    """

    metrics: Dict[str, Any]


class ESGDataMock(BaseModel):
    summary: Dict[str, Any]
    metrics: Dict[str, Any]
    environmentalMetrics: Dict[str, Any]
    socialMetrics: Dict[str, Any]
    governanceMetrics: Dict[str, Any]


class ESGDataResponse(BaseModel):
    mockData: ESGDataMock
    insights: List[str]


# Mini report model with 4 sections
class ESGMiniReport(BaseModel):
    baseline: str
    benchmark: str
    performance_vs_benchmark: str
    ai_recommendations: List[str]


# Monthly history for invoices (last 6 months)
class InvoiceMonthHistory(BaseModel):
    month_label: Optional[str] = None
    energyKWh: Optional[float] = None
    total_current_charges: Optional[float] = None
    total_amount_due: Optional[float] = None
    maximum_demand_kva: Optional[float] = None
    # NEW: estimated carbon emissions for that month, in tCO2e
    carbonTco2e: Optional[float] = None


# Invoice summary model for PDF parsing
class InvoiceSummary(BaseModel):
    filename: str
    company_name: Optional[str] = None
    account_number: Optional[str] = None
    tax_invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    total_current_charges: Optional[float] = None
    total_amount_due: Optional[float] = None
    total_energy_kwh: Optional[float] = None
    categories: List[str] = Field(default_factory=list)
    sixMonthHistory: List[InvoiceMonthHistory] = Field(default_factory=list)
    # base64 PNG (no data: prefix) extracted from invoice if available
    logo_base64: Optional[str] = None


# ================== APP ==================
app = FastAPI(
    title="AfricaESG.AI Backend",
    version="1.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================== GLOBAL STATE ==================
DEFAULT_ESG_INPUT = ESGInput(
    company_name="Company",
    period="FY2024",
    carbon_emissions_tons=18500,
    energy_consumption_mwh=1250,
    water_use_m3=55000,
    waste_generated_tons=180,
    fuel_litres=50000,
    social_score_raw=78,
    governance_score_raw=82,
)


def load_last_esg_from_disk() -> ESGInput:
    """
    Load last_esg_input from disk if available; fall back to DEFAULT_ESG_INPUT.
    """
    if not os.path.exists(LAST_ESG_JSON_PATH):
        return DEFAULT_ESG_INPUT
    try:
        with open(LAST_ESG_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ESGInput(**data)
    except Exception as exc:
        print(f"Failed to load {LAST_ESG_JSON_PATH}: {exc}")
        return DEFAULT_ESG_INPUT


def save_last_esg_to_disk(esg_input: ESGInput) -> None:
    """
    Persist last_esg_input to disk so that after restart / refresh
    we still show the last uploaded / analysed dataset.
    """
    try:
        with open(LAST_ESG_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(esg_input.dict(), f, indent=2)
    except Exception as exc:
        print(f"Failed to save {LAST_ESG_JSON_PATH}: {exc}")


def load_last_esg_rows_from_disk() -> List[Dict[str, Any]]:
    """
    Load the last uploaded ESG rows (for charts) from disk.
    Used to repopulate environmentalMetrics.uploadedRows after restart.
    """
    if not os.path.exists(LAST_ESG_ROWS_PATH):
        return []
    try:
        with open(LAST_ESG_ROWS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception as exc:
        print(f"Failed to load {LAST_ESG_ROWS_PATH}: {exc}")
        return []


def save_last_esg_rows_to_disk(rows: List[Dict[str, Any]]) -> None:
    """
    Persist latest ESG raw rows to disk so charts can be rebuilt after restart.
    """
    try:
        with open(LAST_ESG_ROWS_PATH, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
    except Exception as exc:
        print(f"Failed to save {LAST_ESG_ROWS_PATH}: {exc}")


# Initialize from disk (so last uploaded file is used after restart)
last_esg_input: ESGInput = load_last_esg_from_disk()
last_scores: Optional[ESGScores] = None
last_insights: Optional[ESGInsights] = None

# last uploaded ESG rows (for EnvironmentalCategory charts)
last_esg_uploaded_rows: List[Dict[str, Any]] = load_last_esg_rows_from_disk()

# store latest invoice summaries (single + bulk) for dashboard (in-memory only)
last_invoice_summaries: List[InvoiceSummary] = []

# latest manually uploaded logo (not from invoices)
last_extracted_logo: Optional[str] = None


# ================ LIVE AI (WebSocket) ================
class LiveAIManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass

    async def broadcast(self, message: Dict[str, Any]):
        """
        Send JSON to all connected clients.
        If a connection is broken, remove it.
        """
        dead: List[WebSocket] = []
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except WebSocketDisconnect:
                dead.append(ws)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


live_ai_manager = LiveAIManager()


# ================== ESG HELPER FUNCTIONS ==================
def calculate_esg_scores(esg_input: ESGInput) -> ESGScores:
    """
    Basic scoring logic – adjust as needed.
    Also auto-populates a methodology dictionary that explains how each score was derived.
    """
    # Environmental (lower emissions + lower energy = better)
    e_base = 100 - (esg_input.carbon_emissions_tons / 300)
    e_mod = 100 - (esg_input.energy_consumption_mwh / 25)
    e_score = max(0, min(100, (e_base * 0.7 + e_mod * 0.3)))

    # Social & Governance from raw
    s_score = max(0, min(100, esg_input.social_score_raw))
    g_score = max(0, min(100, esg_input.governance_score_raw))

    overall_score = round((e_score + s_score + g_score) / 3, 1)

    # Auto-generated methodology
    methodology = {
        "environmental": (
            "Computed from carbon_emissions_tons and energy_consumption_mwh. "
            "Lower emissions and lower energy use increase the score. "
            "Formula: 70% weight on emissions intensity, 30% on energy intensity, "
            "with caps at 0 and 100."
        ),
        "social": (
            "Directly based on social_score_raw (0–100), clipped to stay within this range."
        ),
        "governance": (
            "Directly based on governance_score_raw (0–100), clipped to stay within this range."
        ),
        "overall": (
            "Arithmetic mean of E, S and G scores: (E + S + G) / 3, rounded to 1 decimal place."
        ),
    }

    return ESGScores(
        company_name=esg_input.company_name,
        period=esg_input.period,
        e_score=round(e_score, 1),
        s_score=round(s_score, 1),
        g_score=round(g_score, 1),
        overall_score=overall_score,
        methodology=methodology,
    )


# ---------- OpenAI helpers (NON-STRICT: fall back instead of 500) ----------
async def _call_openai_json(system_prompt: str, payload: Any, fallback: Any) -> Any:
    """
    Call OpenAI via Chat Completions and expect a JSON object back.
    If OpenAI is not configured or fails, return the fallback.
    """
    if not openai_client or not OPENAI_API_KEY:
        # No AI configured -> just use fallback
        return fallback

    try:
        completion = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, indent=2),
                },
            ],
        )
        content = completion.choices[0].message.content or ""
        return json.loads(content)
    except Exception as exc:
        print("OpenAI JSON error:", exc)
        return fallback


async def _call_openai_lines(
    system_prompt: str, payload: Any, fallback: List[str]
) -> List[str]:
    """
    Call OpenAI via Chat Completions and return a list of cleaned lines.
    If OpenAI is not configured or fails, return fallback bullets.
    """
    if not openai_client or not OPENAI_API_KEY:
        return fallback

    try:
        completion = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, indent=2),
                },
            ],
        )
        text = completion.choices[0].message.content or ""
        if not text:
            return fallback

        lines_raw = [ln.strip() for ln in text.split("\n") if ln.strip()]

        import re as _re

        cleaned: List[str] = []
        for line in lines_raw:
            line_clean = _re.sub(r"^[-•\d\.\s]+", "", line).strip()
            if line_clean:
                cleaned.append(line_clean)

        return cleaned[:6] or fallback
    except Exception as exc:
        print("OpenAI lines error:", exc)
        return fallback


async def generate_esg_ai_insights(
    esg_input: ESGInput, scores: ESGScores
) -> ESGInsights:
    # Default used if AI is off or returns nothing
    default = ESGInsights(
        overall=(
            "ESG performance is stable with opportunities to improve renewables, "
            "supplier diversity and governance integration into executive scorecards."
        ),
        environmental=[
            "Carbon emissions remain material – prioritise energy efficiency and renewable projects at high-emitting sites.",
            "Energy consumption can be linked to cost and carbon tax exposure for better capital allocation.",
            "Consider a structured decarbonisation roadmap over the next 3–5 years.",
        ],
        social=[
            "Social performance is positive, but supplier diversity can be strengthened with targeted SMME programmes.",
            "Employee engagement should be monitored regularly through pulse surveys.",
            "Community investments would benefit from clearer KPIs and outcome tracking.",
        ],
        governance=[
            "Governance maturity is good – integrate ESG KPIs into Board and EXCO scorecards.",
            "Ensure data privacy, ethics and compliance training are embedded across the organisation.",
            "Align ESG risk registers with enterprise risk management and internal audit.",
        ],
    )

    system_prompt = (
        "You are an ESG reporting advisor supporting IFRS S1/S2 and TCFD-aligned disclosures "
        "for African corporates. Given ESG inputs and scores, produce:\n"
        "- A short overall narrative paragraph\n"
        "- 3 bullet points for Environmental\n"
        "- 3 bullet points for Social\n"
        "- 3 bullet points for Governance\n"
        "Return JSON with keys: overall, environmental, social, governance. "
        "Each of environmental/social/governance should be an array of strings."
    )

    payload = {"input": esg_input.dict(), "scores": scores.dict()}
    parsed = await _call_openai_json(
        system_prompt,
        payload,
        fallback=default.dict(),
    )

    return ESGInsights(
        overall=parsed.get("overall", default.overall),
        environmental=parsed.get("environmental", default.environmental),
        social=parsed.get("social", default.social),
        governance=parsed.get("governance", default.governance),
    )


# Mini report generator
async def generate_esg_mini_report(
    esg_input: ESGInput, scores: ESGScores
) -> ESGMiniReport:
    target_band = 70.0

    default = ESGMiniReport(
        baseline=(
            f"{esg_input.company_name} currently has an overall ESG score of "
            f"{scores.overall_score} (E {scores.e_score}, S {scores.s_score}, "
            f"G {scores.g_score}) for {esg_input.period}."
        ),
        benchmark=(
            "For African corporates with similar size and sector, a typical ESG performance band "
            f"is around {target_band} on a 0–100 scale, with leading peers achieving scores above 80."
        ),
        performance_vs_benchmark=(
            f"The current overall ESG score of {scores.overall_score} places the organisation "
            f"{'above' if scores.overall_score >= target_band else 'below'} the indicative peer benchmark "
            f"of {target_band}. Environmental performance is {scores.e_score}, social {scores.s_score} "
            f"and governance {scores.g_score}, highlighting where targeted interventions could close gaps."
        ),
        ai_recommendations=[
            "Prioritise high-impact decarbonisation levers that reduce both carbon tax exposure and operating costs.",
            "Strengthen social indicators by formalising supplier diversity targets and monitoring workforce inclusion.",
            "Deepen governance maturity by embedding ESG KPIs into executive scorecards and risk management processes.",
            "Define 2–3 flagship ESG initiatives over the next 12–24 months with clear milestones and accountability.",
        ],
    )

    system_prompt = (
        "You are an ESG advisor preparing a concise, board-ready ESG mini report "
        "for an African corporate. Using the provided ESG inputs and scores, "
        "produce a short JSON object with exactly these keys:\n"
        "- baseline: A 2–3 sentence description of the current ESG position.\n"
        "- benchmark: A 2–3 sentence description of a realistic benchmark or peer band.\n"
        "- performance_vs_benchmark: A 2–3 sentence comparison explaining whether the company "
        "is ahead, on par or behind that benchmark, highlighting E/S/G.\n"
        "- ai_recommendations: An array of 3–5 concise, action-oriented recommendations.\n\n"
        "Keep the language neutral, practical and suitable for board and EXCO audiences. "
        "Return ONLY valid JSON."
    )

    payload = {"input": esg_input.dict(), "scores": scores.dict()}
    parsed = await _call_openai_json(
        system_prompt,
        payload,
        fallback=default.dict(),
    )

    return ESGMiniReport(
        baseline=parsed.get("baseline", default.baseline),
        benchmark=parsed.get("benchmark", default.benchmark),
        performance_vs_benchmark=parsed.get(
            "performance_vs_benchmark", default.performance_vs_benchmark
        ),
        ai_recommendations=parsed.get("ai_recommendations", default.ai_recommendations),
    )


# ---------- Structured pillar insights ----------
async def generate_pillar_insights(
    pillar: str, metrics: Dict[str, Any]
) -> List[str]:
    if pillar == "environmental":
        fallback = [
            "Environmental performance baseline reflects current energy use, emissions, waste and fuel consumption derived from your latest ESG dataset.",
            "Comparable African industrial peers typically target steady reductions in energy intensity and emissions over a 3–5 year horizon, with growing use of renewables.",
            "Against this benchmark, your environmental profile shows clear opportunities to improve efficiency, reduce carbon exposure and strengthen waste and fuel management.",
            "Prioritise high-impact efficiency projects at the most energy-intensive sites to reduce both cost and carbon tax exposure.",
            "Investigate key waste streams for reduction, recycling or beneficiation opportunities that support circular economy outcomes.",
            "Assess the role of renewables, PPAs or onsite solar in stabilising long-term energy costs and reducing dependence on carbon-intensive grid power.",
        ]
        system_prompt = (
            "You are an ESG analyst focusing ONLY on Environmental metrics "
            "(energy, emissions, waste, fuel). "
            "Using the provided environmental metrics, produce EXACTLY six short lines "
            "in this order:\n"
            "1) Baseline – a one- to two-sentence description of the current Environmental position.\n"
            "2) Benchmark – a one- to two-sentence description of a realistic peer benchmark or performance band for Environmental.\n"
            "3) Performance vs benchmark – a one- to two-sentence explanation of how the organisation compares to that benchmark.\n"
            "4) Recommendation – a concise, action-oriented Environmental recommendation.\n"
            "5) Recommendation – another concise Environmental recommendation.\n"
            "6) Recommendation – another concise Environmental recommendation.\n\n"
            "Return ONLY these six lines, one per line, with no numbering, bullet symbols or labels."
        )

    elif pillar == "social":
        fallback = [
            "Social performance baseline reflects current supplier diversity, employee engagement and community programme signals from your latest ESG upload.",
            "Comparable African corporates often target steadily improving diversity, engagement and CSI impact as part of their broader ESG journey.",
            "Against this benchmark, your social indicators show a mix of strengths and gaps that can be addressed through focused people, supplier and community initiatives.",
            "Formalise supplier diversity targets, including increased spend with local SMMEs and black-owned suppliers, and track progress quarterly.",
            "Introduce regular, lightweight pulse surveys to monitor employee engagement and identify specific hotspots in teams or sites.",
            "Strengthen community investments by defining clearer CSI outcomes, KPIs and reporting that link to your core business strategy.",
        ]
        system_prompt = (
            "You are an ESG analyst focusing ONLY on Social metrics "
            "(supplier diversity, employee engagement, community, human capital). "
            "Using the provided social metrics, produce EXACTLY six short lines in this order:\n"
            "1) Baseline – a one- to two-sentence description of the current Social position.\n"
            "2) Benchmark – a one- to two-sentence description of a realistic peer benchmark or performance band for Social.\n"
            "3) Performance vs benchmark – a one- to two-sentence explanation of how the organisation compares to that benchmark.\n"
            "4) Recommendation – a concise, action-oriented Social recommendation.\n"
            "5) Recommendation – another concise Social recommendation.\n"
            "6) Recommendation – another concise Social recommendation.\n\n"
            "Return ONLY these six lines, one per line, with no numbering, bullet symbols or labels."
        )

    else:
        fallback = [
            "Governance performance baseline reflects your current governance structures, policies, compliance indicators and supply chain oversight derived from the latest ESG input.",
            "Peer organisations typically aim for progressively higher levels of policy coverage, board oversight and ESG integration into enterprise risk management and internal audit.",
            "Compared to this benchmark, your governance position appears broadly sound but with clear opportunities to deepen ESG integration, data quality and supplier governance.",
            "Embed ESG KPIs into Board and EXCO scorecards to reinforce accountability for sustainability outcomes and risk management.",
            "Ensure that data privacy, ethics and anti-corruption controls are embedded, regularly tested and reported into governance structures.",
            "Extend ESG screening into supplier onboarding and ongoing supply chain management, aligned with enterprise risk and internal audit plans.",
        ]
        system_prompt = (
            "You are an ESG analyst focusing ONLY on Governance metrics "
            "(policies, compliance, ethics, supply chain). "
            "Using the provided governance metrics, produce EXACTLY six short lines in this order:\n"
            "1) Baseline – a one- to two-sentence description of the current Governance position.\n"
            "2) Benchmark – a one- to two-sentence description of a realistic peer benchmark or performance band for Governance.\n"
            "3) Performance vs benchmark – a one- to two-sentence explanation of how the organisation compares to that benchmark.\n"
            "4) Recommendation – a concise, action-oriented Governance recommendation.\n"
            "5) Recommendation – another concise Governance recommendation.\n"
            "6) Recommendation – another concise Governance recommendation.\n\n"
            "Return ONLY these six lines, one per line, with no numbering, bullet symbols or labels."
        )

    return await _call_openai_lines(system_prompt, metrics, fallback=fallback)


# ==== UPDATED CARBON HANDLING: build_environmental_metrics_from_input ====
def build_environmental_metrics_from_input(
    esg_input: ESGInput, scores: ESGScores
) -> Dict[str, Any]:
    """
    Build time-series data for charts, preferring the raw uploaded ESG rows.

    If last_esg_uploaded_rows is available, derive:
      - energyUsage[] (kWh)
      - co2Emissions[] (tCO2e)
      - waste[] (t)
      - fuelUsage[] (L)

    Otherwise, fall back to synthetic series derived from aggregated ESGInput.

    NOTE: when building from uploaded rows we now normalise the series and then
    keep ONLY the LAST 6 entries – so the frontend Carbon chart reflects the
    last 6 months/periods from the uploaded dataset.
    """
    from math import isnan

    global last_esg_uploaded_rows

    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            if v is None:
                return default
            f = float(v)
            if isnan(f):
                return default
            return f
        except Exception:
            return default

    # ---------- 1) Try to build from uploaded ESG rows ----------
    if last_esg_uploaded_rows:
        try:
            df = pd.DataFrame(last_esg_uploaded_rows)

            energy_usage: List[float] = []
            co2_emissions: List[float] = []
            waste: List[float] = []
            fuel_usage: List[float] = []

            # ENERGY (kWh): prefer "Electricity (kWh)", else "Energy (kWh)"
            energy_col = None
            for candidate in ["Electricity (kWh)", "Energy (kWh)"]:
                if candidate in df.columns:
                    energy_col = candidate
                    break
            if energy_col:
                energy_usage = (
                    df[energy_col].fillna(0).apply(_safe_float).round().tolist()
                )

            # WASTE (t) from "Waste Generated (kg)"
            if "Waste Generated (kg)" in df.columns:
                waste = (
                    (
                        df["Waste Generated (kg)"]
                        .fillna(0)
                        .apply(_safe_float) / 1000.0
                    )
                    .round(2)
                    .tolist()
                )

            # FUEL (L)
            if "Fuel (L)" in df.columns:
                fuel_usage = (
                    df["Fuel (L)"].fillna(0).apply(_safe_float).round().tolist()
                )

            # CO₂ (t): robust detection from columns, else compute from energy + fuel
            co2_col = None

            # 1) Exact candidates first
            exact_candidates = ["CO2 (t)", "Carbon Emissions (t)", "Emissions (tCO2e)"]
            for candidate in exact_candidates:
                if candidate in df.columns:
                    co2_col = candidate
                    break

            # 2) If still nothing, do a fuzzy search for any CO2 / carbon column
            if co2_col is None:
                for col_name in df.columns:
                    name = str(col_name).lower()
                    if "co2" in name or "co₂" in name or "carbon" in name:
                        co2_col = col_name
                        break

            if co2_col is not None:
                co2_emissions = (
                    df[co2_col].fillna(0).apply(_safe_float).round(2).tolist()
                )
            else:
                # 3) Fallback: compute from energy + fuel if possible
                if energy_col or "Fuel (L)" in df.columns:
                    EF_ELECTRICITY_T_PER_KWH = 0.0009
                    EF_FUEL_T_PER_L = 0.0027

                    energy_series = (
                        df[energy_col].fillna(0).apply(_safe_float)
                        if energy_col
                        else pd.Series([0.0] * len(df))
                    )
                    fuel_series = (
                        df["Fuel (L)"].fillna(0).apply(_safe_float)
                        if "Fuel (L)" in df.columns
                        else pd.Series([0.0] * len(df))
                    )

                    co2_emissions = (
                        energy_series * EF_ELECTRICITY_T_PER_KWH
                        + fuel_series * EF_FUEL_T_PER_L
                    ).round(2).tolist()

            # If *any* list is non-empty, normalise lengths and return
            if any(
                len(lst) > 0 for lst in [energy_usage, co2_emissions, waste, fuel_usage]
            ):
                max_len = max(
                    len(lst)
                    for lst in [energy_usage, co2_emissions, waste, fuel_usage]
                )

                def _pad(lst: List[float]) -> List[float]:
                    if len(lst) == 0:
                        return [0.0] * max_len
                    if len(lst) < max_len:
                        return lst + [0.0] * (max_len - len(lst))
                    return lst

                # Normalise all series to the same length
                energy_usage = _pad(energy_usage)
                co2_emissions = _pad(co2_emissions)
                waste = _pad(waste)
                fuel_usage = _pad(fuel_usage)

                # Keep ONLY the last 6 periods (most recent 6 rows)
                if max_len > 6:
                    start = max_len - 6
                    energy_usage = energy_usage[start:]
                    co2_emissions = co2_emissions[start:]
                    waste = waste[start:]
                    fuel_usage = fuel_usage[start:]

                return {
                    "energyUsage": energy_usage,
                    "co2Emissions": co2_emissions,
                    "waste": waste,
                    "fuelUsage": fuel_usage,
                }

        except Exception as exc:
            # If anything fails here, don't crash – just log and fall back
            print("Failed to build environmental metrics from uploaded rows:", exc)

    # ---------- 2) FALLBACK: synthetic series from aggregated ESGInput ----------
    periods = 6
    base_energy_kwh = esg_input.energy_consumption_mwh * 1000
    base_co2 = esg_input.carbon_emissions_tons
    total_fuel_l = getattr(esg_input, "fuel_litres", 0.0) or 0.0

    energy_usage: List[float] = []
    co2_emissions: List[float] = []
    waste: List[float] = []
    fuel_usage: List[float] = []

    for i in range(periods):
        factor = 0.8 + 0.05 * i
        energy_usage.append(round(base_energy_kwh / periods * factor))
        co2_emissions.append(round(base_co2 / periods * factor, 1))
        waste.append(
            round(esg_input.waste_generated_tons / periods * (0.9 + 0.02 * i), 2)
        )

        fuel_factor = 0.9 + 0.02 * i
        fuel_usage.append(round(total_fuel_l / periods * fuel_factor))

    return {
        "energyUsage": energy_usage,
        "co2Emissions": co2_emissions,
        "waste": waste,
        "fuelUsage": fuel_usage,
    }


def build_social_metrics_from_input(
    esg_input: ESGInput, scores: ESGScores
) -> Dict[str, Any]:
    supplier_diversity = round(esg_input.social_score_raw * 0.25)
    employee_engagement = round(esg_input.social_score_raw)
    community_programs = 40
    return {
        "supplierDiversity": supplier_diversity,
        "employeeEngagement": employee_engagement,
        "communityPrograms": community_programs,
    }


def build_governance_metrics_from_input(
    esg_input: ESGInput, scores: ESGScores
) -> Dict[str, Any]:
    corporate_governance = "Strong" if scores.g_score >= 80 else "Developing"
    iso_compliance = "ISO 9001 Certified" if scores.g_score >= 75 else "In progress"
    business_ethics = "High" if scores.g_score >= 85 else "Moderate"
    return {
        "corporateGovernance": corporate_governance,
        "dataPrivacy": "Compliant",
        "isoCompliance": iso_compliance,
        "businessEthics": business_ethics,
        "codeOfEthics": "Yes",
        "informationSecurityPolicy": "Yes",
        "supplierSustainabilityCompliance": 72,
        "supplierAuditsCompleted": 58,
        "supplierEsgCompliance": "Medium",
        "totalGovernanceTrainings": 24,
        "totalEnvironmentalTrainings": 18,
        "totalComplianceFindings": 1 if scores.g_score >= 80 else 3,
    }


def build_summary_and_metrics(esg_input: ESGInput, scores: ESGScores):
    """
    Build summary + metrics blocks used by the dashboard.

    RenewableEnergyShare:
      * If ESG upload has explicit renewable kWh columns, compute:
        renewables_kwh / total_kwh * 100
      * Otherwise fall back to a proxy derived from the E-score.

    Carbon Tax / Tax Allowances / Carbon Credits / Energy Savings:
      * Prefer invoice-derived energy and CO₂ baselines if invoices exist.
      * Fall back to ESGInput-based values if no invoices are available.
    """
    from math import isnan

    global last_esg_uploaded_rows, last_invoice_summaries

    # ---------- helper for safe numeric ----------
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            if v is None:
                return default
            f = float(v)
            if isnan(f):
                return default
            return f
        except Exception:
            return default

    # ---------- base pillar metrics ----------
    env_metrics = build_environmental_metrics_from_input(esg_input, scores)
    soc_metrics = build_social_metrics_from_input(esg_input, scores)
    gov_metrics = build_governance_metrics_from_input(esg_input, scores)

    # ---------- TOTAL ENERGY (kWh) from ESG upload ----------
    total_energy = 0.0

    if last_esg_uploaded_rows:
        try:
            df_rows = pd.DataFrame(last_esg_uploaded_rows)
            energy_col = None
            for candidate in ["Electricity (kWh)", "Energy (kWh)"]:
                if candidate in df_rows.columns:
                    energy_col = candidate
                    break
            if energy_col:
                total_energy = float(
                    df_rows[energy_col].fillna(0).apply(_safe_float).sum()
                )
        except Exception as exc:
            print("Failed to compute total_energy from uploaded rows:", exc)

    # Fallback: sum of chart energyUsage series
    if not total_energy:
        total_energy = float(sum(env_metrics.get("energyUsage", [])))

    # Fuel (L) from chart metrics
    total_fuel = float(sum(env_metrics.get("fuelUsage", [])))

    # ---------- Prefer invoice-derived baselines if available ----------
    invoice_aggregated = None
    if last_invoice_summaries:
        try:
            recent_invoices = list(last_invoice_summaries)
            # sort newest -> oldest by invoice_date
            recent_invoices.sort(
                key=lambda x: _parse_invoice_date_string(x.invoice_date)
                or datetime.min,
                reverse=True,
            )
            recent_invoices = recent_invoices[:6]
            invoice_aggregated = aggregate_invoice_environmental_metrics(
                recent_invoices
            )
        except Exception as exc:
            print(
                "Failed to aggregate invoice metrics inside build_summary_and_metrics:",
                exc,
            )

    if invoice_aggregated:
        # Override with invoice-derived baselines
        total_energy = invoice_aggregated.get("total_energy_kwh", total_energy) or 0.0
        carbon_for_metrics = (
            invoice_aggregated.get(
                "estimated_co2_tonnes", esg_input.carbon_emissions_tons
            )
            or 0.0
        )
    else:
        # Fall back to ESGInput-based emissions
        carbon_for_metrics = esg_input.carbon_emissions_tons

    # ---------- Renewables (%) ----------
    renewable_share: float

    # If we have row-level ESG data and a renewables column, calculate a real share
    if last_esg_uploaded_rows and total_energy > 0:
        try:
            df_rows = pd.DataFrame(last_esg_uploaded_rows)

            renewable_candidates = [
                "Renewable (kWh)",
                "Renewables (kWh)",
                "Solar (kWh)",
                "Wind (kWh)",
                "Hydro (kWh)",
            ]

            renewable_cols = [c for c in renewable_candidates if c in df_rows.columns]

            if renewable_cols:
                total_renewables = 0.0
                for c in renewable_cols:
                    total_renewables += float(
                        df_rows[c].fillna(0).apply(_safe_float).sum()
                    )
                if total_renewables > 0:
                    renewable_share = max(
                        0.0,
                        min(100.0, round((total_renewables / total_energy) * 100, 1)),
                    )
                else:
                    # no actual renewable energy, explicit 0%
                    renewable_share = 0.0
            else:
                # no explicit renewable column, fall back to score-based proxy
                renewable_share = max(10, min(60, round(scores.e_score / 1.5)))
        except Exception as exc:
            print("Failed to compute renewable share from uploaded rows:", exc)
            renewable_share = max(10, min(60, round(scores.e_score / 1.5)))
    else:
        # No uploaded rows or no energy – use proxy from the E-score
        renewable_share = max(10, min(60, round(scores.e_score / 1.5)))

    # ---------- Summary blocks ----------
    summary = {
        "environmental": {
            "totalEnergyConsumption": total_energy,  # kWh (invoice-preferred)
            "totalFuelUsageLitres": total_fuel,  # L
            "renewableEnergyShare": renewable_share,  # %
            "carbonEmissions": carbon_for_metrics,  # tCO₂e (invoice-preferred)
        },
        "social": {
            "supplierDiversity": soc_metrics["supplierDiversity"],
            "customerSatisfaction": scores.s_score,
            "humanCapital": round((scores.s_score + scores.overall_score) / 2),
        },
        "governance": {
            "corporateGovernance": gov_metrics["corporateGovernance"],
            "iso9001Compliance": gov_metrics["isoCompliance"],
            "businessEthics": gov_metrics["businessEthics"],
            "totalGovernanceTrainings": gov_metrics["totalGovernanceTrainings"],
            "totalEnvironmentalTrainings": gov_metrics["totalEnvironmentalTrainings"],
            "totalComplianceFindings": gov_metrics["totalComplianceFindings"],
        },
    }

    # ---------- Financial / carbon KPIs ----------
    CARBON_TAX_RATE = 1500.0  # R per tCO₂e (placeholder)
    TAX_ALLOWANCE_FACTOR = 0.30  # 30% of carbon tax
    CARBON_CREDIT_FACTOR = 0.15  # 15% of emissions as proxy
    ENERGY_SAVINGS_FACTOR = 0.12  # 12% of energy as proxy savings

    metrics = {
        # Carbon Tax (R) = emissions (tCO₂e) × tax rate
        "carbonTax": round(carbon_for_metrics * CARBON_TAX_RATE),
        # Applicable Tax Allowances (R) – assumed 30% of carbon tax
        "taxAllowances": round(
            carbon_for_metrics * CARBON_TAX_RATE * TAX_ALLOWANCE_FACTOR
        ),
        # Carbon Credits Generated (t) – assumed 15% of emissions as a proxy
        "carbonCredits": round(carbon_for_metrics * CARBON_CREDIT_FACTOR),
        # Energy Savings (kWh) – assumed 12% of total energy as a proxy
        "energySavings": round(total_energy * ENERGY_SAVINGS_FACTOR),
    }

    return summary, metrics, env_metrics, soc_metrics, gov_metrics


# ==== UPDATED CARBON HANDLING: build_esg_input_from_excel ====
def build_esg_input_from_excel(content: bytes) -> Tuple[ESGInput, List[Dict[str, Any]]]:
    """
    Excel -> (ESGInput, uploaded_rows) converter.
    uploaded_rows is used by the frontend charts as metrics.uploadedRows.
    """
    try:
        df = pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read Excel file: {exc}",
        )

    def col(name: str):
        return df[name] if name in df.columns else None

    def col_sum(name: str) -> float:
        s = col(name)
        return float(s.fillna(0).sum()) if s is not None else 0.0

    def col_mean(name: str, default: float = 0.0) -> float:
        s = col(name)
        return float(s.fillna(default).mean()) if s is not None else default

    # ENVIRONMENTAL
    elec_kwh = col_sum("Electricity (kWh)")
    energy_mwh = elec_kwh / 1000.0

    fuel_l = col_sum("Fuel (L)")

    water_kl = col_sum("Water (kl)")
    water_m3 = water_kl

    waste_kg = col_sum("Waste Generated (kg)")
    waste_tons = waste_kg / 1000.0

    # Robust carbon detection
    carbon = 0.0

    # 1) Exact candidates
    exact_carbon_cols = ["CO2 (t)", "Carbon Emissions (t)", "Emissions (tCO2e)"]
    for candidate in exact_carbon_cols:
        if candidate in df.columns:
            carbon = col_sum(candidate)
            break

    # 2) Fuzzy match any CO2 / carbon column if still zero
    if carbon == 0.0:
        fuzzy_col = None
        for c in df.columns:
            name = str(c).lower()
            if "co2" in name or "co₂" in name or "carbon" in name:
                fuzzy_col = c
                break
        if fuzzy_col is not None:
            carbon = col_sum(fuzzy_col)

    # 3) Fallback to energy + fuel if still zero
    if carbon == 0.0:
        EF_ELECTRICITY_T_PER_KWH = 0.0009
        EF_FUEL_T_PER_L = 0.0027
        carbon = elec_kwh * EF_ELECTRICITY_T_PER_KWH + fuel_l * EF_FUEL_T_PER_L

    # SOCIAL
    employees_series = col("Employees")
    training_hours_series = col("Training Hours")
    incidents_series = col("Incidents")

    total_employees = (
        float(employees_series.fillna(0).sum())
        if employees_series is not None
        else 0.0
    )
    total_training_hours = (
        float(training_hours_series.fillna(0).sum())
        if training_hours_series is not None
        else 0.0
    )
    total_incidents = (
        float(incidents_series.fillna(0).sum()) if incidents_series is not None else 0.0
    )
    n_rows = len(df) if len(df) > 0 else 1

    women_avg = col_mean("Women (%)", default=30.0)
    youth_avg = col_mean("Youth (%)", default=20.0)

    if total_employees > 0:
        training_per_employee = total_training_hours / total_employees
    else:
        training_per_employee = 0.0

    diversity_component = (women_avg + youth_avg) / 2
    diversity_component = max(0.0, min(diversity_component * 0.7, 70.0))

    training_component = max(0.0, min(training_per_employee * 2.0, 20.0))

    incidents_per_period = total_incidents / n_rows
    incident_penalty = max(0.0, min(incidents_per_period * 4.0, 20.0))

    social_score_raw = diversity_component + training_component - incident_penalty
    social_score_raw = max(0.0, min(social_score_raw, 100.0))

    # GOVERNANCE
    gov_trainings_avg = col_mean("Governance Trainings", default=5.0)
    governance_score_raw = max(0.0, min(gov_trainings_avg * 5.0, 100.0))

    esg_input = ESGInput(
        company_name="Excel Import",
        period="Imported (aggregated from Excel)",
        carbon_emissions_tons=carbon or DEFAULT_ESG_INPUT.carbon_emissions_tons,
        energy_consumption_mwh=energy_mwh or DEFAULT_ESG_INPUT.energy_consumption_mwh,
        water_use_m3=water_m3 or DEFAULT_ESG_INPUT.water_use_m3,
        waste_generated_tons=waste_tons or DEFAULT_ESG_INPUT.waste_generated_tons,
        fuel_litres=fuel_l or DEFAULT_ESG_INPUT.fuel_litres,
        social_score_raw=social_score_raw or DEFAULT_ESG_INPUT.social_score_raw,
        governance_score_raw=governance_score_raw
        or DEFAULT_ESG_INPUT.governance_score_raw,
    )

    # Raw rows for charts
    uploaded_rows: List[Dict[str, Any]] = df.to_dict(orient="records")

    return esg_input, uploaded_rows


# --- Invoice helpers ---
def detect_invoice_categories(text: str) -> List[str]:
    """
    Look at the invoice text and tag which ESG categories appear:
    Energy, Carbon, Water, Waste, Fuel.
    """
    t = text.lower()
    cats: Set[str] = set()

    # Energy: electricity, kWh, kVA, demand, TOU, etc.
    if any(
        w in t
        for w in [
            "electricity",
            "kwh",
            "kva",
            "time of use",
            "tou",
            "active energy",
            "bulk electricity",
        ]
    ):
        cats.add("Energy")
        cats.add("Carbon")

    # Water
    if any(w in t for w in ["water (kl)", "water kl", "water ", "sewer", "sewage"]):
        cats.add("Water")

    # Waste
    if any(w in t for w in ["waste", "refuse removal", "landfill"]):
        cats.add("Waste")

    # Fuel
    if any(
        w in t
        for w in ["diesel", "petrol", " fuel", "fuel ", "coal ", "coal invoice"]
    ):
        cats.add("Fuel")

    return sorted(cats)


def extract_company_name(text: str) -> Optional[str]:
    """
    Try to detect the company name from invoice text.
    Looks for known patterns and generic uppercase headers.
    """
    # Example: THE DUBE TRADEPORT CORPORATION
    m = re.search(r"(DUBE TRADEPORT.+?CORPORATION)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().title()

    # Generic: top lines in all caps (likely company name)
    for line in text.split("\n")[:15]:
        cleaned = line.strip()
        if cleaned.isupper() and len(cleaned) > 6:
            return cleaned.title()

    return None


# ---------- Previous usage history parsing ----------
MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def _parse_month_label(label: str) -> Optional[datetime]:
    """
    Convert labels like 'Oct-24' to datetime(2024, 10, 1).
    """
    m = re.match(r"([A-Za-z]{3})-(\d{2})", label.strip())
    if not m:
        return None
    mon_str, yy = m.groups()
    mon = MONTH_MAP.get(mon_str.title())
    if not mon:
        return None
    year = 2000 + int(yy)
    try:
        return datetime(year, mon, 1)
    except ValueError:
        return None


def parse_previous_usage_history(full_text: str) -> List[InvoiceMonthHistory]:
    """
    Parse the 'Previous ... Usage' table into a 6-month history.

    Many invoices show 12 months (e.g. Oct-24 ... Sep-25); we want the
    LAST 6 months in chronological order (e.g. Apr-25 ... Sep-25).
    """

    # 1) Isolate the block between "Previous ... Usage" and the next section
    block_match = re.search(
        r"Previous\s+\d+\s+Usage(.*?)(Maximum Demand / Notified Demand|Page\s+\d+\s+of\s+\d+|$)",
        full_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not block_match:
        return []

    block = block_match.group(1)
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return []

    # 2) Get ALL month labels in order (e.g. Oct-24 ... Sep-25)
    month_labels: List[str] = re.findall(r"\b([A-Z][a-z]{2}-\d{2})\b", block)
    if len(month_labels) < 2:
        return []

    # 3) Collect ALL Energy (kWh) values and ALL Rands values
    energy_vals: List[float] = []
    rand_vals: List[float] = []

    for ln in lines:
        lower = ln.lower()

        if lower.startswith("energy (kwh)"):
            nums = re.findall(r"[0-9,]+(?:\.\d+)?", ln)
            for tok in nums:
                try:
                    energy_vals.append(float(tok.replace(",", "")))
                except ValueError:
                    continue

        elif lower.startswith("rands"):
            nums = re.findall(r"[0-9,]+(?:\.\d+)?", ln)
            for tok in nums:
                try:
                    rand_vals.append(float(tok.replace(",", "")))
                except ValueError:
                    continue

    # 4) Match up month, energy, rands
    n = min(len(month_labels), len(energy_vals), len(rand_vals))
    if n == 0:
        return []

    entries = []
    for i in range(n):
        label = month_labels[i]
        energy = energy_vals[i]
        rands = rand_vals[i]
        entries.append((label, energy, rands))

    # 5) Sort by date (oldest → newest) and keep the LAST 6
    def _key(item):
        dt = _parse_month_label(item[0])
        return dt or datetime.min

    entries.sort(key=_key)
    last_six = entries[-6:]  # latest 6 months

    # 6) Convert to InvoiceMonthHistory models, including per-month carbon
    EF_ELECTRICITY_T_PER_KWH = 0.0009  # same factor as elsewhere

    history: List[InvoiceMonthHistory] = []
    for label, energy, rands in last_six:
        carbon = None
        if energy is not None:
            try:
                carbon = round(float(energy) * EF_ELECTRICITY_T_PER_KWH, 2)
            except Exception:
                carbon = None

        history.append(
            InvoiceMonthHistory(
                month_label=label,
                energyKWh=energy,
                total_current_charges=rands,
                total_amount_due=None,
                maximum_demand_kva=None,
                carbonTco2e=carbon,
            )
        )

    return history


def _parse_invoice_date_string(date_str: Optional[str]) -> Optional[datetime]:
    """
    Convert invoice_date string to datetime.
    Supports common formats like:
    - 2025/10/01
    - 01/10/2025
    - 25/10/01  (YY/MM/DD)
    """
    if not date_str:
        return None

    s = date_str.strip()
    for fmt in ("%Y/%m/%d", "%d/%m/%Y", "%y/%m/%d", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


# ---------- helper to extract long numeric IDs ----------
def extract_long_numeric(value: Optional[str]) -> Optional[str]:
    """
    Extract the longest numeric chunk (6+ digits) from a string.
    Used to pull things like 1611033830025 out of 'VAT Number Guarantee 1611033830025'.
    """
    if not value:
        return None
    matches = re.findall(r"\d{6,}", value)
    if not matches:
        return None
    matches.sort(key=len, reverse=True)
    return matches[0]


# ---------- invoice environmental aggregation + AI ----------
def aggregate_invoice_environmental_metrics(
    invoices: List[InvoiceSummary],
) -> Dict[str, Any]:
    """
    Aggregate invoice-level energy and cost metrics for AI analysis.
    Typically used on the last N invoices.
    """
    if not invoices:
        return {
            "invoice_count": 0,
            "total_energy_kwh": 0.0,
            "total_current_charges": 0.0,
            "total_amount_due": 0.0,
            "avg_energy_kwh": 0.0,
            "avg_current_charges": 0.0,
            "blended_tariff_r_per_kwh": 0.0,
            "estimated_co2_tonnes": 0.0,
        }

    def _safe(v: Optional[float]) -> float:
        if v is None:
            return 0.0
        try:
            return float(v)
        except Exception:
            return 0.0

    invoice_count = len(invoices)
    total_energy_kwh = sum(_safe(inv.total_energy_kwh) for inv in invoices)
    total_current_charges = sum(_safe(inv.total_current_charges) for inv in invoices)
    total_amount_due = sum(_safe(inv.total_amount_due) for inv in invoices)

    avg_energy_kwh = total_energy_kwh / invoice_count if invoice_count > 0 else 0.0
    avg_current_charges = (
        total_current_charges / invoice_count if invoice_count > 0 else 0.0
    )

    blended_tariff = (
        total_current_charges / total_energy_kwh if total_energy_kwh > 0 else 0.0
    )

    # simple grid factor for South Africa – 0.0009 tCO2e per kWh
    EF_ELECTRICITY_T_PER_KWH = 0.0009
    estimated_co2_tonnes = total_energy_kwh * EF_ELECTRICITY_T_PER_KWH

    return {
        "invoice_count": invoice_count,
        "total_energy_kwh": round(total_energy_kwh, 2),
        "total_current_charges": round(total_current_charges, 2),
        "total_amount_due": round(total_amount_due, 2),
        "avg_energy_kwh": round(avg_energy_kwh, 2),
        "avg_current_charges": round(avg_current_charges, 2),
        "blended_tariff_r_per_kwh": round(blended_tariff, 4),
        "estimated_co2_tonnes": round(estimated_co2_tonnes, 2),
    }


async def generate_invoice_environmental_insights(
    aggregated: Dict[str, Any],
    invoices: List[InvoiceSummary],
) -> List[str]:
    """
    Use OpenAI to generate short, board-friendly Environmental insights
    based on bulk invoice energy + cost metrics.
    Falls back to deterministic text if AI is unavailable.
    """
    inv_count = aggregated.get("invoice_count", 0)
    total_kwh = aggregated.get("total_energy_kwh", 0.0)
    total_charges = aggregated.get("total_current_charges", 0.0)
    blended_tariff = aggregated.get("blended_tariff_r_per_kwh", 0.0)
    est_co2 = aggregated.get("estimated_co2_tonnes", 0.0)

    fallback = [
        (
            f"Over the last {inv_count} electricity invoices, total energy "
            f"consumption was {total_kwh:,.0f} kWh with current charges of approximately "
            f"R {total_charges:,.2f}, providing a clear recent energy cost baseline."
        ),
        (
            f"The implied blended tariff is around R {blended_tariff:,.2f} per kWh, "
            f"and this consumption equates to roughly {est_co2:,.0f} tCO₂e using a "
            "typical grid emission factor for South Africa."
        ),
        (
            "Monthly consumption appears relatively stable with no strong downward trend, "
            "suggesting that structural efficiency improvements have not yet significantly "
            "shifted the overall load profile."
        ),
        (
            "A small number of high-cost months likely contribute disproportionately to overall spend, "
            "which warrants deeper analysis of peak demand, time-of-use charges and penalties."
        ),
        (
            "Short-term optimisation should focus on bill validation, tariff optimisation "
            "and targeting the highest-use months and sites for detailed operational diagnostics."
        ),
        (
            "These invoice-derived energy and emissions baselines should be integrated into "
            "your ESG dashboard to track carbon tax exposure, progress against reduction "
            "targets and compliance with IFRS S2 / TCFD-style disclosures."
        ),
    ]

    system_prompt = (
        "You are an ESG and energy analyst for African corporates. "
        "You receive a JSON object with:\n"
        "- aggregated invoice metrics (energy_kwh, current_charges, blended tariff, estimated_co2_tonnes, etc.)\n"
        "- a list of individual invoice summaries (company_name, invoice_date, total_energy_kwh, total_current_charges, etc.).\n\n"
        "Using ONLY this data, produce EXACTLY six short lines:\n"
        "1) Baseline – concise description of recent energy and cost profile from the invoices.\n"
        "2) Tariff & carbon – concise view of blended tariff and implied carbon exposure.\n"
        "3) Pattern – what the usage and cost pattern over the period suggests (trend/flat/volatile).\n"
        "4) Cost concentration – comment on high-cost months / risk areas.\n"
        "5) Optimisation levers – practical actions to reduce energy cost and emissions.\n"
        "6) Next step – how to link these invoice insights into broader ESG / IFRS S2 reporting.\n\n"
        "Return ONLY these six lines, one per line, with no numbering, bullets or labels."
    )

    payload = {
        "aggregated": aggregated,
        "invoices": [inv.dict() for inv in invoices],
    }

    return await _call_openai_lines(system_prompt, payload, fallback=fallback)


# ================== LOGO EXTRACTION ==================
def extract_logo_from_pdf(pdf_content: bytes) -> Optional[str]:
    """
    Extract logo from PDF using PyMuPDF.
    Returns a base64 PNG data (without 'data:image/png;base64,' prefix),
    or None if nothing suitable is found.
    """
    if fitz is None:
        # PyMuPDF not available
        return None

    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        images: List[Dict[str, Any]] = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)

            for img in image_list:
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)

                    # Only handle RGB / grayscale, skip CMYK etc.
                    if pix.n < 5:
                        img_data = pix.tobytes("png")
                        pil_img = Image.open(io.BytesIO(img_data))

                        width, height = pil_img.size
                        area = width * height
                        aspect_ratio = width / height if height > 0 else 1.0

                        # Heuristic: reasonable size and aspect ratio for a logo
                        is_logo_candidate = (
                            50 <= width <= 500
                            and 50 <= height <= 500
                            and 0.3 <= aspect_ratio <= 3.0
                            and area >= 2500  # at least 50x50
                        )

                        if is_logo_candidate:
                            buffered = io.BytesIO()
                            pil_img.save(buffered, format="PNG")
                            img_str = base64.b64encode(
                                buffered.getvalue()
                            ).decode("utf-8")

                            images.append(
                                {
                                    "image": pil_img,
                                    "base64": img_str,
                                    "width": width,
                                    "height": height,
                                    "area": area,
                                    "aspect_ratio": aspect_ratio,
                                    "page": page_num,
                                }
                            )
                    pix = None  # free
                except Exception:
                    continue

        doc.close()

        if not images:
            return None

        # Prefer square-ish, medium-size images on earlier pages
        images.sort(
            key=lambda x: (
                abs(x["aspect_ratio"] - 1.0),     # closer to square
                abs(x["area"] - 10000),           # around 100x100
                x["page"],                        # earlier page first
            )
        )

        return images[0]["base64"] if images else None

    except Exception as e:
        print(f"PDF logo extraction error: {e}")
        return None


def extract_logo_from_image(image_content: bytes) -> Optional[str]:
    """
    Extract logo from a direct image upload (PNG, JPG, etc.).
    Returns a base64 PNG string (without data: prefix).
    """
    try:
        img = Image.open(io.BytesIO(image_content))

        # Convert paletted/transparent images to RGB on white background
        if img.mode in ("RGBA", "LA", "P"):
            if img.mode == "P":
                img = img.convert("RGBA")

            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "RGBA":
                background.paste(img, mask=img.split()[3])
            else:
                # For 'LA' etc. use second channel as mask if exists
                channels = img.split()
                mask = channels[1] if len(channels) > 1 else None
                background.paste(img, mask=mask)

            img = background

        # Resize if too big, keeping aspect ratio
        max_size = (300, 300)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    except Exception as e:
        print(f"Image logo extraction error: {e}")
        return None


def parse_invoice_pdf(content: bytes, filename: str) -> InvoiceSummary:
    """
    Parse a single invoice PDF into an InvoiceSummary.

    - Extracts full text from all pages
    - Tries to detect company name, account number, invoice/tax numbers, dates
    - Estimates total_current_charges / total_amount_due and total_energy_kwh
    - Tags ESG categories and parses 6-month usage history where present
    - Attempts to extract a logo image as base64 PNG
    """
    if PdfReader is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "PDF parsing is not available. Install 'pypdf' "
                "and restart the backend."
            ),
        )

    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to open PDF {filename}: {exc}",
        )

    # ---- Extract raw text from all pages ----
    full_text_parts: List[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt:
            full_text_parts.append(txt)

    full_text = "\n".join(full_text_parts)

    # ---- Basic fields ----
    company = extract_company_name(full_text)
    categories = detect_invoice_categories(full_text)
    six_month_history = parse_previous_usage_history(full_text)

    # ---------- Account / VAT / Tax invoice identifiers ----------
    account_number: Optional[str] = None
    m = re.search(
        r"Account\s*Number\s*[:\-]?\s*([A-Za-z0-9/\- ]+)",
        full_text,
        flags=re.IGNORECASE,
    )
    if m:
        account_number = m.group(1).strip()

    # Many invoices show: "VAT Number Guarantee  1611033830025"
    vat_line: Optional[str] = None
    m_vat = re.search(
        r"VAT\s+Number\s+Guarantee.*",
        full_text,
        flags=re.IGNORECASE,
    )
    if m_vat:
        vat_line = m_vat.group(0).strip()
        if account_number is None:
            acc_from_vat = extract_long_numeric(vat_line)
            if acc_from_vat:
                account_number = acc_from_vat

    # Helper: detect if we only have the label and no real number
    def _looks_like_label_only(s: str) -> bool:
        s_clean = s.strip()
        if re.search(r"\d", s_clean):
            return False
        return bool(
            re.match(r"(?i)^(tax\s+invoice(\s+no\.?)?|invoice(\s+no\.?)?)$", s_clean)
        )

    # Tax / invoice label: capture the tail of the line
    tax_invoice_number_raw: Optional[str] = None
    m = re.search(
        r"(Tax\s+Invoice|Invoice)\s*(No\.?|Number)?\s*[:\-]?\s*([A-Za-z0-9/\- ]{1,80})",
        full_text,
        flags=re.IGNORECASE,
    )
    if m:
        tax_invoice_number_raw = m.group(3).strip()

    # Build candidate strings for numeric extraction
    id_candidates: List[str] = []
    if tax_invoice_number_raw:
        id_candidates.append(tax_invoice_number_raw)
    if account_number:
        id_candidates.append(account_number)
    if vat_line:
        id_candidates.append(vat_line)

    tax_invoice_number: Optional[str] = None

    # Prefer a long numeric ID (>= 6 digits) from any candidate
    for cand in id_candidates:
        num = extract_long_numeric(cand)
        if num:
            tax_invoice_number = num
            break

    # Extra: if tax_invoice_number_raw is just "Tax Invoice No" / "Invoice No", ignore it
    if tax_invoice_number is None and tax_invoice_number_raw:
        if _looks_like_label_only(tax_invoice_number_raw):
            tax_invoice_number_raw = None

    # Try to find digits *after* the "Tax Invoice No" label, even across line breaks
    if tax_invoice_number is None:
        m_label_num = re.search(
            r"Tax\s+Invoice\s*No\.?\s*[:\-]?\s*([\d\s]{6,})",
            full_text,
            flags=re.IGNORECASE,
        )
        if m_label_num:
            num = extract_long_numeric(m_label_num.group(1))
            if num:
                tax_invoice_number = num

    # As a last resort, take the longest numeric chunk from the whole doc (>= 8 digits)
    if tax_invoice_number is None:
        all_long_nums = re.findall(r"\d{8,}", full_text)
        if all_long_nums:
            all_long_nums.sort(key=len, reverse=True)
            tax_invoice_number = all_long_nums[0]

    # If we STILL have nothing, fall back to any non-empty textual candidate
    if tax_invoice_number is None:
        for cand in id_candidates:
            if cand and not _looks_like_label_only(cand):
                tax_invoice_number = cand.strip()
                break

    # ---- Dates ----
    invoice_date = None
    m = re.search(
        r"Invoice\s*Date\s*[:\-]?\s*([0-9]{2,4}/[0-9]{1,2}/[0-9]{2,4})",
        full_text,
        flags=re.IGNORECASE,
    )
    if m:
        invoice_date = m.group(1).strip()

    due_date = None
    m = re.search(
        r"Due\s*Date\s*[:\-]?\s*([0-9]{2,4}/[0-9]{1,2}/[0-9]{2,4})",
        full_text,
        flags=re.IGNORECASE,
    )
    if m:
        due_date = m.group(1).strip()

    # ---- Monetary amounts ----
    def _find_amount(patterns: List[str]) -> Optional[float]:
        for pat in patterns:
            m_inner = re.search(
                pat + r".*?([0-9][0-9.,]*)",
                full_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if m_inner:
                raw = m_inner.group(1)
                try:
                    return float(raw.replace(",", ""))
                except ValueError:
                    continue
        return None

    total_current_charges = _find_amount(
        [
            r"Total\s+Current\s+Charges",
            r"Current\s+Charges",
        ]
    )

    total_amount_due = _find_amount(
        [
            r"Total\s+Amount\s+Due",
            r"Amount\s+Due",
            r"Total\s+Due",
        ]
    )

    # Fallback if only one was found
    if total_current_charges is None and total_amount_due is not None:
        total_current_charges = total_amount_due

    # ---- Energy (kWh) ----
    total_energy_kwh = None
    m = re.search(
        r"(Total\s+Energy|Energy\s*\(kWh\)).*?([0-9][0-9.,]*)",
        full_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        raw = m.group(2)
        try:
            total_energy_kwh = float(raw.replace(",", ""))
        except ValueError:
            total_energy_kwh = None

    # ---- Logo extraction (non-fatal) ----
    logo_base64: Optional[str] = None
    try:
        logo_base64 = extract_logo_from_pdf(content)
    except Exception as logo_error:
        print(f"Logo extraction failed for {filename}: {logo_error}")

    return InvoiceSummary(
        filename=filename,
        company_name=company,
        account_number=account_number,
        tax_invoice_number=tax_invoice_number,
        invoice_date=invoice_date,
        due_date=due_date,
        total_current_charges=total_current_charges,
        total_amount_due=total_amount_due,
        total_energy_kwh=total_energy_kwh,
        categories=categories,
        sixMonthHistory=six_month_history,
        logo_base64=logo_base64,
    )


# ---------- LIVE SNAPSHOT HELPERS ----------
async def build_live_esg_snapshot() -> Dict[str, Any]:
    """
    Build a single JSON payload that the frontend can use for
    'live AI' updates over WebSocket.

    It bundles:
    - latest ESG scores
    - summary + metrics
    - pillar metrics
    - combined AI ESG insights
    - latest invoice environmental metrics + insights (if invoices exist)
    """
    global last_insights

    esg_input = last_esg_input or DEFAULT_ESG_INPUT
    scores = last_scores or calculate_esg_scores(esg_input)

    # core ESG summary + KPIs
    summary, metrics, env_metrics, soc_metrics, gov_metrics = build_summary_and_metrics(
        esg_input, scores
    )

    # main ESG AI insights (re-use last_insights if already generated)
    if last_insights is None:
        last_insights = await generate_esg_ai_insights(esg_input, scores)

    insights_obj = last_insights

    combined_insights: List[str] = [
        insights_obj.overall,
        *insights_obj.environmental,
        *insights_obj.social,
        *insights_obj.governance,
    ]

    # attach uploaded ESG rows for charts
    env_metrics_with_rows = dict(env_metrics)
    if last_esg_uploaded_rows:
        env_metrics_with_rows["uploadedRows"] = last_esg_uploaded_rows

    # optional: invoice-derived env metrics + AI
    invoice_metrics: Optional[Dict[str, Any]] = None
    invoice_insights: Optional[List[str]] = None
    if last_invoice_summaries:
        invoices_sorted = list(last_invoice_summaries)
        invoices_sorted.sort(
            key=lambda x: _parse_invoice_date_string(x.invoice_date) or datetime.min,
            reverse=True,
        )
        recent = invoices_sorted[:6]
        invoice_metrics = aggregate_invoice_environmental_metrics(recent)
        invoice_insights = await generate_invoice_environmental_insights(
            invoice_metrics, recent
        )

    payload: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "esgInput": esg_input.dict(),
        "scores": scores.dict(),
        "summary": summary,
        "metrics": metrics,
        "environmentalMetrics": env_metrics_with_rows,
        "socialMetrics": soc_metrics,
        "governanceMetrics": gov_metrics,
        "esgInsights": combined_insights,
    }

    if invoice_metrics is not None:
        payload["invoiceEnvironmentalMetrics"] = invoice_metrics
    if invoice_insights is not None:
        payload["invoiceEnvironmentalInsights"] = invoice_insights

    return payload


async def push_live_ai_update():
    """
    If any clients are connected on /ws/live-ai,
    send them the latest snapshot.
    """
    if not live_ai_manager.active_connections:
        return
    snapshot = await build_live_esg_snapshot()
    await live_ai_manager.broadcast({"type": "live-esg-update", "data": snapshot})


# ================== ROUTES ==================
@app.get("/api/health", tags=["System"])
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get(
    "/platform/overview",
    response_model=PlatformOverview,
    tags=["Platform"],
)
async def platform_overview():
    """
    Used by Dashboard.jsx to populate platformStats.
    """
    return PlatformOverview(
        countries_supported=50,
        esg_reports_generated=10000,
        compliance_accuracy=0.99,
        ai_support_mode="24/7",
    )


@app.post(
    "/esg/analyse",
    response_model=AnalyseResponse,
    tags=["ESG Analysis"],
)
async def esg_analyse(payload: ESGInput):
    """
    Main analysis endpoint used by Dashboard.jsx.
    """
    global last_esg_input, last_scores, last_insights

    scores = calculate_esg_scores(payload)
    insights = await generate_esg_ai_insights(payload, scores)

    last_esg_input = payload
    last_scores = scores
    last_insights = insights

    save_last_esg_to_disk(last_esg_input)

    # LIVE: push snapshot to any connected dashboards
    await push_live_ai_update()

    return AnalyseResponse(scores=scores, insights=insights)


@app.get(
    "/api/esg-data",
    response_model=ESGDataResponse,
    tags=["AI Insights"],
)
async def api_esg_data():
    """
    ESG data bundle: summary + metrics + pillar metric blocks + combined insights.
    Always built from last_esg_input (loaded from disk on startup).
    """
    esg_input = last_esg_input or DEFAULT_ESG_INPUT
    scores = last_scores or calculate_esg_scores(esg_input)

    summary, metrics, env_metrics, soc_metrics, gov_metrics = build_summary_and_metrics(
        esg_input, scores
    )

    if last_insights:
        combined_insights: List[str] = [
            last_insights.overall,
            *last_insights.environmental,
            *last_insights.social,
            *last_insights.governance,
        ]
    else:
        combined_insights = [
            "Initial ESG summary generated from latest ESG input.",
        ]

    # Attach last uploaded rows (if any) for charts
    env_metrics_with_rows = dict(env_metrics)
    if last_esg_uploaded_rows:
        env_metrics_with_rows["uploadedRows"] = last_esg_uploaded_rows

    mock = ESGDataMock(
        summary=summary,
        metrics=metrics,
        environmentalMetrics=env_metrics_with_rows,
        socialMetrics=soc_metrics,
        governanceMetrics=gov_metrics,
    )

    return ESGDataResponse(mockData=mock, insights=combined_insights)


@app.post(
    "/api/esg-upload",
    response_model=ESGDataResponse,
    tags=["AI Insights"],
)
async def api_esg_upload(file: UploadFile = File(...)):
    """
    Upload ESG data (JSON or Excel), update the current ESG run and
    return mockData + insights similar to /api/esg-data`.

    Supported:
    - .json with ESGInput shape (optionally including uploadedRows/rows/data)
    - .xlsx / .xls with ESG columns.
    """
    global last_esg_input, last_scores, last_insights, last_esg_uploaded_rows

    filename = (file.filename or "").lower()

    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {exc}",
        )

    uploaded_rows: List[Dict[str, Any]] = []

    if filename.endswith(".json"):
        try:
            payload = json.loads(content.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON file: {exc}",
            )

        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400,
                detail="JSON root must be an object.",
            )

        uploaded_rows = (
            payload.get("uploadedRows")
            or payload.get("rows")
            or payload.get("data")
            or []
        )
        if not isinstance(uploaded_rows, list):
            uploaded_rows = []

        clean_payload = {
            k: v
            for k, v in payload.items()
            if k not in ["uploadedRows", "rows", "data"]
        }

        try:
            esg_input = ESGInput(**clean_payload)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"JSON does not match ESGInput schema: {exc}",
            )

    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        esg_input, uploaded_rows = build_esg_input_from_excel(content)
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload .json, .xlsx or .xls.",
        )

    scores = calculate_esg_scores(esg_input)
    insights_obj = await generate_esg_ai_insights(esg_input, scores)

    last_esg_input = esg_input
    last_scores = scores
    last_insights = insights_obj
    last_esg_uploaded_rows = uploaded_rows or []

    save_last_esg_to_disk(last_esg_input)
    save_last_esg_rows_to_disk(last_esg_uploaded_rows)

    summary, metrics, env_metrics, soc_metrics, gov_metrics = build_summary_and_metrics(
        esg_input, scores
    )

    env_metrics_with_rows = dict(env_metrics)
    if last_esg_uploaded_rows:
        env_metrics_with_rows["uploadedRows"] = last_esg_uploaded_rows

    combined_insights: List[str] = [
        insights_obj.overall,
        *insights_obj.environmental,
        *insights_obj.social,
        *insights_obj.governance,
    ]

    mock = ESGDataMock(
        summary=summary,
        metrics=metrics,
        environmentalMetrics=env_metrics_with_rows,
        socialMetrics=soc_metrics,
        governanceMetrics=gov_metrics,
    )

    # LIVE: ESG dataset changed -> push snapshot
    await push_live_ai_update()

    return ESGDataResponse(mockData=mock, insights=combined_insights)


@app.get(
    "/api/esg-mini-report",
    response_model=ESGMiniReport,
    tags=["AI Insights"],
)
async def api_esg_mini_report():
    """
    Returns a compact, AI-generated ESG mini report with:
    1) Baseline
    2) Benchmark
    3) Performance vs benchmark
    4) AI Recommendations[]
    Always uses the latest ESG dataset (last_esg_input).
    """
    esg_input = last_esg_input or DEFAULT_ESG_INPUT
    scores = last_scores or calculate_esg_scores(esg_input)
    mini = await generate_esg_mini_report(esg_input, scores)
    return mini


@app.get(
    "/api/environmental-insights",
    response_model=PillarInsightsResponse,
    tags=["AI Insights"],
)
async def api_environmental_insights():
    """
    For EnvironmentalCategory.jsx:
    returns environmentalMetrics + LIVE AI insights (or fallback text).
    Includes uploadedRows from the latest ESG upload so charts can be built
    directly from the raw data.
    """
    esg_input = last_esg_input or DEFAULT_ESG_INPUT
    scores = last_scores or calculate_esg_scores(esg_input)
    env_metrics = build_environmental_metrics_from_input(esg_input, scores)
    insights = await generate_pillar_insights("environmental", env_metrics)

    env_metrics_with_rows = dict(env_metrics)
    if last_esg_uploaded_rows:
        env_metrics_with_rows["uploadedRows"] = last_esg_uploaded_rows

    return PillarInsightsResponse(metrics=env_metrics_with_rows, insights=insights)


@app.get(
    "/api/social-insights",
    response_model=PillarInsightsResponse,
    tags=["AI Insights"],
)
async def api_social_insights_get():
    """
    GET /api/social-insights

    Used by SimulationContext (and any other simple consumer) to:
    - Build social metrics from the latest ESG input
    - Return metrics + AI-generated insights
    """
    esg_input = last_esg_input or DEFAULT_ESG_INPUT
    scores = last_scores or calculate_esg_scores(esg_input)

    soc_metrics = build_social_metrics_from_input(esg_input, scores)
    insights = await generate_pillar_insights("social", soc_metrics)

    return PillarInsightsResponse(metrics=soc_metrics, insights=insights)


@app.post(
    "/api/social-insights",
    response_model=PillarInsightsResponse,
    tags=["AI Insights"],
)
async def api_social_insights_post(payload: SocialInsightsRequest):
    """
    POST /api/social-insights

    Used by SocialCategory.jsx when it sends current social metrics:

    Body:
    {
      "metrics": {
        "supplierDiversity": ...,
        "employeeEngagement": ...,
        "communityPrograms": ...,
        ...
      }
    }

    - Uses the provided metrics directly for AI analysis.
    - If metrics is empty, falls back to backend-built metrics from last_esg_input.
    """
    esg_input = last_esg_input or DEFAULT_ESG_INPUT
    scores = last_scores or calculate_esg_scores(esg_input)

    incoming_metrics = payload.metrics or {}

    if not incoming_metrics:
        soc_metrics = build_social_metrics_from_input(esg_input, scores)
    else:
        soc_metrics = incoming_metrics

    insights = await generate_pillar_insights("social", soc_metrics)

    return PillarInsightsResponse(metrics=soc_metrics, insights=insights)


@app.get(
    "/api/governance-insights",
    response_model=PillarInsightsResponse,
    tags=["AI Insights"],
)
async def api_governance_insights():
    """
    For GovernanceCategory.jsx:
    returns governanceMetrics + LIVE governanceInsights (or fallback text).
    """
    esg_input = last_esg_input or DEFAULT_ESG_INPUT
    scores = last_scores or calculate_esg_scores(esg_input)
    gov_metrics = build_governance_metrics_from_input(esg_input, scores)
    insights = await generate_pillar_insights("governance", gov_metrics)
    return PillarInsightsResponse(metrics=gov_metrics, insights=insights)


# ================== LOGO ROUTES ==================
@app.get("/api/company-logo", tags=["Logo"])
async def get_company_logo():
    """
    Get the latest extracted company logo from uploaded invoices
    or from the last manually uploaded logo.
    """
    global last_invoice_summaries, last_extracted_logo, last_esg_input

    # Prefer the most recent invoice that has a logo and company name
    if last_invoice_summaries:
        for invoice in last_invoice_summaries:
            if invoice.logo_base64 and invoice.company_name:
                return {
                    "success": True,
                    "company_name": invoice.company_name,
                    "logo": f"data:image/png;base64,{invoice.logo_base64}",
                    "source": invoice.filename,
                }

    # Fallback: manually uploaded logo
    if last_extracted_logo:
        company_name = last_esg_input.company_name if last_esg_input else "Company"
        return {
            "success": True,
            "company_name": company_name,
            "logo": f"data:image/png;base64,{last_extracted_logo}",
            "source": "manual-upload",
        }

    raise HTTPException(
        status_code=404,
        detail="No logo found in uploaded invoices or manual uploads",
    )


@app.post("/api/upload-logo", tags=["Logo"])
async def upload_logo_directly(file: UploadFile = File(...)):
    """
    Upload a logo image directly (not from invoice).
    Supports PDF (extract first suitable image) or common image formats.
    """
    global last_extracted_logo

    filename = (file.filename or "").lower()

    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded logo file: {exc}",
        )

    logo_base64: Optional[str] = None

    if filename.endswith(".pdf"):
        logo_base64 = extract_logo_from_pdf(content)
    elif filename.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")):
        logo_base64 = extract_logo_from_image(content)
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload PDF or image.",
        )

    if not logo_base64:
        raise HTTPException(
            status_code=400,
            detail="No logo could be extracted from the file",
        )

    # Store in global for later use
    last_extracted_logo = logo_base64

    return {
        "success": True,
        "logo": f"data:image/png;base64,{logo_base64}",
        "message": "Logo uploaded successfully",
    }


# ---------- INVOICE ROUTES (single, bulk, list) ----------
@app.post(
    "/api/invoice-upload",
    response_model=InvoiceSummary,
    tags=["Invoices"],
)
async def api_invoice_upload(file: UploadFile = File(...)):
    """
    Upload a single invoice PDF and return a structured summary,
    including company_name for DashboardCategory / EnvironmentalCategory.
    """
    global last_invoice_summaries

    filename = file.filename or "invoice.pdf"
    lower_name = filename.lower()
    if not lower_name.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a .pdf invoice file.",
        )

    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file {filename}: {exc}",
        )

    summary = parse_invoice_pdf(content, filename)

    # store for dashboard (latest first)
    last_invoice_summaries.insert(0, summary)
    if len(last_invoice_summaries) > 200:
        last_invoice_summaries.pop()

    # LIVE: invoice changed -> push update
    await push_live_ai_update()

    return summary


@app.post(
    "/api/invoice-bulk-upload",
    response_model=List[InvoiceSummary],
    tags=["Invoices"],
)
async def api_invoice_bulk_upload(files: List[UploadFile] = File(...)):
    """
    Upload multiple invoice PDFs, parse each and return a list of summaries,
    including company_name and sixMonthHistory for each invoice.
    """
    global last_invoice_summaries

    if not files:
        raise HTTPException(status_code=400, detail="No PDF files uploaded.")

    results: List[InvoiceSummary] = []

    for f in files:
        filename = f.filename or "invoice.pdf"
        lower_name = filename.lower()
        if not lower_name.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for {filename}. PDFs only.",
            )

        try:
            content = await f.read()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read uploaded file {filename}: {exc}",
            )

        summary = parse_invoice_pdf(content, filename)
        results.append(summary)

    # prepend for dashboard and limit size
    last_invoice_summaries = results + last_invoice_summaries
    if len(last_invoice_summaries) > 200:
        last_invoice_summaries = last_invoice_summaries[:200]

    # LIVE: invoices changed -> push update
    await push_live_ai_update()

    return results


@app.get(
    "/api/invoices",
    response_model=List[InvoiceSummary],
    tags=["Invoices"],
)
async def api_invoices(last_months: Optional[int] = None):
    """
    Return invoice summaries.

    - If last_months is provided (e.g. ?last_months=6), only invoices with
      invoice_date in that rolling window are returned.
    - Otherwise, all invoices in memory are returned.
    """
    if not last_invoice_summaries:
        return []

    # If no filter requested, just return whatever we have
    if last_months is None:
        return last_invoice_summaries

    try:
        months = int(last_months)
    except (TypeError, ValueError):
        months = None

    if not months or months <= 0:
        return last_invoice_summaries

    # Compute cutoff date = first of the month (now - last_months)
    now = datetime.utcnow()
    cutoff = now.replace(day=1)
    month = cutoff.month - months
    year = cutoff.year
    while month <= 0:
        month += 12
        year -= 1
    cutoff = cutoff.replace(year=year, month=month)

    filtered: List[InvoiceSummary] = []
    for inv in last_invoice_summaries:
        dt = _parse_invoice_date_string(inv.invoice_date)
        if dt and dt >= cutoff:
            filtered.append(inv)

    # Sort newest -> oldest by invoice_date
    filtered.sort(
        key=lambda x: _parse_invoice_date_string(x.invoice_date) or datetime.min,
        reverse=True,
    )

    return filtered


@app.get(
    "/api/invoice-environmental-insights",
    response_model=PillarInsightsResponse,
    tags=["Invoices"],
)
async def api_invoice_environmental_insights(
    last_n: int = Query(6, ge=1, le=50),
):
    """
    Live AI Environmental insights based on the most recent N invoices.

    - Aggregates energy (kWh) and cost from invoice summaries
    - Calls OpenAI to generate short board-ready insights
    - Returns { metrics: {...}, insights: [ ... ] } just like other pillar endpoints
    """
    if not last_invoice_summaries:
        raise HTTPException(
            status_code=404,
            detail="No invoices uploaded yet for analysis.",
        )

    invoices_sorted = list(last_invoice_summaries)
    invoices_sorted.sort(
        key=lambda x: _parse_invoice_date_string(x.invoice_date) or datetime.min,
        reverse=True,
    )
    recent = invoices_sorted[:last_n]

    aggregated = aggregate_invoice_environmental_metrics(recent)
    insights = await generate_invoice_environmental_insights(aggregated, recent)

    return PillarInsightsResponse(metrics=aggregated, insights=insights)


# ---------- LIVE AI WEBSOCKET ----------
@app.websocket("/ws/live-ai")
async def websocket_live_ai(websocket: WebSocket):
    """
    Live AI WebSocket.

    Frontend can connect to:
      ws://<host>/ws/live-ai   or   wss://<host>/ws/live-ai

    It will receive:
    - an initial ESG + invoice + AI snapshot
    - a new snapshot every time ESG / invoices are updated
    - on-demand 'refresh' if it sends 'refresh' or 'ping' text
    """
    await live_ai_manager.connect(websocket)
    try:
        # send initial snapshot immediately
        initial_payload = await build_live_esg_snapshot()
        await websocket.send_json({"type": "live-esg-update", "data": initial_payload})

        while True:
            msg = await websocket.receive_text()
            if msg.strip().lower() in ("refresh", "ping"):
                snapshot = await build_live_esg_snapshot()
                await websocket.send_json(
                    {"type": "live-esg-update", "data": snapshot}
                )
    except WebSocketDisconnect:
        live_ai_manager.disconnect(websocket)
    except Exception:
        live_ai_manager.disconnect(websocket)
