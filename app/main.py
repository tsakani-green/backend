import os
import io
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
import base64
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, WebSocket
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator                           
from PIL import Image
import asyncio

load_dotenv()

# ================== CONFIG ==================
# Carbon calculation constants
CARBON_EMISSION_FACTOR = 0.85  # kg CO2 per kWh
CARBON_FORMULA = "Estimated Carbon (kg CO₂) = Energy (kWh) × 0.85"

# ================== PYDANTIC MODELS ==================
class ESGInput(BaseModel):
    company_name: str = "Example Corp"
    industry: str = "Manufacturing"
    region: str = "Africa"
    energy_consumption_mwh: float = 500.0
    renewable_energy_percentage: float = 25.0
    water_usage_m3: float = 10000.0
    waste_generated_tons: float = 50.0
    carbon_emissions_tons: float = 250.0
    employee_count: int = 1000
    female_leadership_percentage: float = 30.0
    training_hours_per_employee: float = 20.0
    board_independence_percentage: float = 60.0
    ethics_training_completion: float = 85.0
    supplier_screening_percentage: float = 75.0

class ESGScores(BaseModel):
    e_score: float = Field(..., ge=0, le=100)
    s_score: float = Field(..., ge=0, le=100)
    g_score: float = Field(..., ge=0, le=100)
    overall_score: float = Field(..., ge=0, le=100)
    calculated_at: Optional[datetime] = None
    
    @validator('calculated_at', pre=True, always=True)
    def set_calculated_at(cls, v):
        return v or datetime.now()

class ESGInsights(BaseModel):
    overall: str
    environmental: List[str]
    social: List[str]
    governance: List[str]
    generated_at: Optional[datetime] = None
    
    @validator('generated_at', pre=True, always=True)
    def set_generated_at(cls, v):
        return v or datetime.now()

class AnalyseResponse(BaseModel):
    scores: ESGScores
    insights: ESGInsights
    analysed_at: Optional[datetime] = None
    
    @validator('analysed_at', pre=True, always=True)
    def set_analysed_at(cls, v):
        return v or datetime.now()

class ESGDataMock(BaseModel):
    summary: Dict[str, Any]
    metrics: Dict[str, Any]
    environmentalMetrics: Dict[str, Any]
    socialMetrics: Dict[str, Any]
    governanceMetrics: Dict[str, Any]
    last_updated: Optional[datetime] = None
    
    @validator('last_updated', pre=True, always=True)
    def set_last_updated(cls, v):
        return v or datetime.now()

class ESGDataResponse(BaseModel):
    mockData: ESGDataMock
    insights: List[str]

class InvoiceMonthHistory(BaseModel):
    month_label: Optional[str] = None
    energyKWh: Optional[float] = None
    total_current_charges: Optional[float] = None
    total_amount_due: Optional[float] = None
    maximum_demand_kva: Optional[float] = None
    carbonTco2e: Optional[float] = None

class InvoiceSummary(BaseModel):
    filename: str
    company_name: str
    invoice_date: str
    total_energy_kwh: Optional[float] = None
    total_current_charges: Optional[float] = None
    total_amount_due: Optional[float] = None
    sixMonthHistory: List[InvoiceMonthHistory] = []
    
    @validator('sixMonthHistory', pre=True)
    def sort_and_calculate_six_month(cls, v):
        """Sort history and ensure we have exactly 6 months"""
        if not v:
            return []
        
        # Sort by month label
        sorted_history = sorted(
            v,
            key=lambda x: x.get('month_label', '')
        )
        
        # Take latest 6 months
        return sorted_history[-6:]

class WebSocketMessage(BaseModel):
    type: str
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.now()

# ================== DEFAULT VALUES ==================
DEFAULT_ESG_INPUT = ESGInput()

# ================== FASTAPI APP ==================
app = FastAPI(title="ESG Analysis API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== GLOBAL STATE ==================
last_esg_input: ESGInput = DEFAULT_ESG_INPUT
last_scores: Optional[ESGScores] = None
last_insights: Optional[ESGInsights] = None
last_esg_uploaded_rows: List[Dict[str, Any]] = []
last_invoice_summaries: List[InvoiceSummary] = []
last_extracted_logo: Optional[str] = None

# ================== HELPER FUNCTIONS ==================
def build_esg_input_from_excel(content: bytes) -> Tuple[ESGInput, List[Dict[str, Any]]]:
    """
    Build ESGInput from Excel file.
    """
    try:
        # Try reading as Excel
        df = pd.read_excel(io.BytesIO(content))
        
        # Convert to list of dictionaries for rows
        rows = df.to_dict(orient='records')
        
        # Extract metrics for ESGInput
        esg_input = ESGInput(
            company_name=df.get('Company Name', ['Unknown'])[0] if 'Company Name' in df.columns else "Unknown",
            industry=df.get('Industry', ['Manufacturing'])[0] if 'Industry' in df.columns else "Manufacturing",
            energy_consumption_mwh=float(df['Energy (MWh)'].mean()) if 'Energy (MWh)' in df.columns else 500.0,
            renewable_energy_percentage=float(df['Renewable %'].mean()) if 'Renewable %' in df.columns else 25.0,
            carbon_emissions_tons=float(df['Carbon Emissions (t)'].mean()) if 'Carbon Emissions (t)' in df.columns else 250.0,
        )
        
        return esg_input, rows
    except Exception as e:
        print(f"Error parsing Excel: {e}")
        # Return default if parsing fails
        return ESGInput(), []

def calculate_esg_scores(esg_input: ESGInput) -> ESGScores:
    """
    Calculate E, S, G scores from the ESGInput data.
    """
    # Environmental score (weighted average)
    e_score = (
        (100 - min(esg_input.energy_consumption_mwh / 1000 * 10, 100)) * 0.4 +
        esg_input.renewable_energy_percentage * 0.3 +
        (100 - min(esg_input.water_usage_m3 / 1000, 100)) * 0.2 +
        (100 - min(esg_input.waste_generated_tons * 2, 100)) * 0.1
    )
    
    # Social score
    s_score = (
        esg_input.female_leadership_percentage * 0.3 +
        min(esg_input.training_hours_per_employee * 2, 100) * 0.4 +
        esg_input.supplier_screening_percentage * 0.3
    )
    
    # Governance score
    g_score = (
        esg_input.board_independence_percentage * 0.4 +
        esg_input.ethics_training_completion * 0.4 +
        min(esg_input.employee_count / 10, 100) * 0.2
    )
    
    # Overall score (weighted average)
    overall_score = e_score * 0.4 + s_score * 0.3 + g_score * 0.3
    
    return ESGScores(
        e_score=round(min(max(e_score, 0), 100), 1),
        s_score=round(min(max(s_score, 0), 100), 1),
        g_score=round(min(max(g_score, 0), 100), 1),
        overall_score=round(min(max(overall_score, 0), 100), 1),
        calculated_at=datetime.now()
    )

async def generate_esg_ai_insights(esg_input: ESGInput, scores: ESGScores) -> ESGInsights:
    """
    Generate AI insights for ESG analysis.
    """
    # Mock implementation
    overall = f"Company {esg_input.company_name} shows {'strong' if scores.overall_score > 70 else 'moderate' if scores.overall_score > 50 else 'needs improvement'} ESG performance with overall score of {scores.overall_score}/100."
    
    environmental = [
        f"Environmental score: {scores.e_score}/100. Renewable energy at {esg_input.renewable_energy_percentage}%.",
        f"Energy consumption: {esg_input.energy_consumption_mwh} MWh annually.",
        f"Carbon emissions calculated using formula: {CARBON_FORMULA}."
    ]
    
    social = [
        f"Social score: {scores.s_score}/100. Female leadership: {esg_input.female_leadership_percentage}%.",
        f"Training: {esg_input.training_hours_per_employee} hours per employee annually.",
        f"Supplier screening: {esg_input.supplier_screening_percentage}% of suppliers screened."
    ]
    
    governance = [
        f"Governance score: {scores.g_score}/100. Board independence: {esg_input.board_independence_percentage}%.",
        f"Ethics training completion: {esg_input.ethics_training_completion}%.",
        f"Employee count: {esg_input.employee_count}."
    ]
    
    return ESGInsights(
        overall=overall,
        environmental=environmental,
        social=social,
        governance=governance,
        generated_at=datetime.now()
    )

def build_environmental_metrics_from_input(
    esg_input: ESGInput, scores: ESGScores
) -> Dict[str, Any]:
    """
    Build time-series data for charts, preferring the raw uploaded ESG rows.
    Uses carbon formula: Estimated Carbon (kg CO₂) = Energy (kWh) × 0.85
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

            # CO₂ (kg): robust detection from columns, else compute from energy
            co2_col = None

            # 1) Exact candidates first
            exact_carbon_cols = ["CO2 (kg)", "Carbon Emissions (kg)", "Emissions (kg CO2)"]
            for candidate in exact_carbon_cols:
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
                # 3) Fallback: compute from energy using formula: Energy (kWh) × 0.85
                if energy_col:
                    energy_series = (
                        df[energy_col].fillna(0).apply(_safe_float)
                        if energy_col
                        else pd.Series([0.0] * len(df))
                    )
                    co2_emissions = (
                        energy_series * CARBON_EMISSION_FACTOR
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
                    "carbon_factor": CARBON_EMISSION_FACTOR,
                    "carbon_formula": CARBON_FORMULA,
                }

        except Exception as exc:
            # If anything fails here, don't crash – just log and fall back
            print("Failed to build environmental metrics from uploaded rows:", exc)

    # ---------- 2) FALLBACK: synthetic series from aggregated ESGInput ----------
    periods = 6
    base_energy_kwh = esg_input.energy_consumption_mwh * 1000
    base_co2 = base_energy_kwh * CARBON_EMISSION_FACTOR  # Using formula: kWh × 0.85
    total_fuel_l = getattr(esg_input, "fuel_litres", 0.0) or 0.0

    energy_usage: List[float] = []
    co2_emissions: List[float] = []
    waste: List[float] = []
    fuel_usage: List[float] = []

    for i in range(periods):
        factor = 0.8 + 0.05 * i
        energy_usage.append(round(base_energy_kwh / periods * factor))
        co2_emissions.append(round(base_co2 / periods * factor, 2))
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
        "carbon_factor": CARBON_EMISSION_FACTOR,
        "carbon_formula": CARBON_FORMULA,
    }

def build_social_metrics_from_input(
    esg_input: ESGInput, scores: ESGScores
) -> Dict[str, Any]:
    """
    Build social metrics for the dashboard.
    """
    periods = 6
    base_supplier = esg_input.supplier_screening_percentage

    supplier_diversity: List[float] = []
    customer_satisfaction: List[float] = []
    human_capital: List[float] = []

    for i in range(periods):
        # Supplier diversity trends upward
        supplier_diversity.append(
            round(base_supplier * (0.95 + 0.01 * i), 1)
        )

        # Customer satisfaction (somewhat stable)
        customer_satisfaction.append(
            round(scores.s_score * (0.98 + 0.005 * i), 1)
        )

        # Human capital (slow growth)
        human_capital.append(
            round(scores.s_score * (0.9 + 0.02 * i), 1)
        )

    return {
        "supplierDiversity": supplier_diversity,
        "customerSatisfaction": customer_satisfaction,
        "humanCapital": human_capital,
        "supplierDiversityValue": round(base_supplier, 1),
        "customerSatisfactionValue": round(scores.s_score, 1),
        "humanCapitalValue": round(scores.s_score, 1),
    }

def build_governance_metrics_from_input(
    esg_input: ESGInput, scores: ESGScores
) -> Dict[str, Any]:
    """
    Build governance metrics for the dashboard.
    """
    periods = 6
    base_compliance = min(scores.g_score + 10, 100)
    base_ethics = esg_input.ethics_training_completion

    corporate_governance: List[float] = []
    iso_compliance: List[float] = []
    business_ethics: List[float] = []

    for i in range(periods):
        # Corporate governance (stable to improving)
        corporate_governance.append(
            round(scores.g_score * (0.98 + 0.005 * i), 1)
        )

        # ISO compliance (slow improvement)
        iso_compliance.append(
            round(base_compliance * (0.95 + 0.01 * i), 1)
        )

        # Business ethics (slow improvement)
        business_ethics.append(
            round(base_ethics * (0.96 + 0.008 * i), 1)
        )

    # Training & compliance totals (mock values)
    total_gov_trainings = esg_input.employee_count * 2  # mock
    total_env_trainings = esg_input.employee_count * 3  # mock
    total_compliance_findings = max(0, 100 - scores.g_score)  # mock

    return {
        "corporateGovernance": corporate_governance,
        "isoCompliance": iso_compliance,
        "businessEthics": business_ethics,
        "corporateGovernanceValue": round(scores.g_score, 1),
        "isoComplianceValue": round(base_compliance, 1),
        "businessEthicsValue": round(base_ethics, 1),
        "totalGovernanceTrainings": int(total_gov_trainings),
        "totalEnvironmentalTrainings": int(total_env_trainings),
        "totalComplianceFindings": int(total_compliance_findings),
    }

def build_summary_and_metrics(esg_input: ESGInput, scores: ESGScores):
    """
    Build summary + metrics blocks used by the dashboard.
    Uses carbon formula: Estimated Carbon (kg CO₂) = Energy (kWh) × 0.85
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

    # ---------- Carbon calculations ----------
    # Calculate carbon using formula: Energy (kWh) × 0.85
    carbon_for_metrics = total_energy * CARBON_EMISSION_FACTOR

    # ---------- Renewables (%) ----------
    renewable_share: float

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
            "totalEnergyConsumption": total_energy,  # kWh
            "totalFuelUsageLitres": total_fuel,  # L
            "renewableEnergyShare": renewable_share,  # %
            "carbonEmissions": round(carbon_for_metrics, 2),  # kg CO₂
            "carbonFormula": CARBON_FORMULA,
        },
        "social": {
            "supplierDiversity": soc_metrics["supplierDiversityValue"],
            "customerSatisfaction": scores.s_score,
            "humanCapital": round((scores.s_score + scores.overall_score) / 2),
        },
        "governance": {
            "corporateGovernance": gov_metrics["corporateGovernanceValue"],
            "iso9001Compliance": gov_metrics["isoComplianceValue"],
            "businessEthics": gov_metrics["businessEthicsValue"],
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
        "carbonTax": round((carbon_for_metrics / 1000) * CARBON_TAX_RATE),
        # Total Charges (R) – placeholder based on energy cost
        "totalCharges": round(total_energy * 2.5),
        # Applicable Tax Allowances (R) – assumed 30% of carbon tax
        "taxAllowances": round(
            (carbon_for_metrics / 1000) * CARBON_TAX_RATE * TAX_ALLOWANCE_FACTOR
        ),
        # Carbon Credits Generated (t) – assumed 15% of emissions as a proxy
        "carbonCredits": round((carbon_for_metrics / 1000) * CARBON_CREDIT_FACTOR),
        # Energy Savings (kWh) – assumed 12% of total energy as a proxy
        "energySavings": round(total_energy * ENERGY_SAVINGS_FACTOR),
        # Carbon Formula
        "carbonFormula": CARBON_FORMULA,
    }

    return summary, metrics, env_metrics, soc_metrics, gov_metrics

# ================== WEBSOCKET MANAGEMENT ==================
active_connections: List[WebSocket] = []

@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo received messages (optional)
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def push_live_ai_update():
    """
    Push a snapshot of the current ESG state to all connected dashboards.
    """
    global last_esg_input, last_scores, last_insights
    
    if not last_esg_input or not last_scores:
        return
    
    snapshot = {
        "type": "esg_snapshot",
        "timestamp": datetime.now().isoformat(),
        "input": last_esg_input.dict() if hasattr(last_esg_input, 'dict') else {},
        "scores": last_scores.dict() if last_scores else {},
        "carbon_formula": CARBON_FORMULA
    }
    
    message = WebSocketMessage(type="esg_update", data=snapshot)
    
    for connection in active_connections:
        try:
            await connection.send_json(message.dict())
        except Exception as e:
            print(f"Failed to send WebSocket update: {e}")

# ================== ROUTES ==================
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "ESG Analysis API", 
        "version": "1.0.0",
        "carbon_formula": CARBON_FORMULA
    }

@app.get("/api/esg-data", tags=["ESG Analysis"])
async def api_esg_data():
    """
    Return the current ESG state (input + scores + insights) as a mock data structure.
    """
    global last_esg_input, last_scores, last_insights
    
    if not last_esg_input or not last_scores:
        # No analysis run yet; compute from default
        last_scores = calculate_esg_scores(DEFAULT_ESG_INPUT)
        last_insights = await generate_esg_ai_insights(DEFAULT_ESG_INPUT, last_scores)
    
    summary, metrics, env_metrics, soc_metrics, gov_metrics = build_summary_and_metrics(
        last_esg_input, last_scores
    )
    
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
    
    combined_insights: List[str] = []
    if last_insights:
        combined_insights = [
            last_insights.overall,
            *last_insights.environmental,
            *last_insights.social,
            *last_insights.governance,
        ]
    
    return ESGDataResponse(mockData=mock, insights=combined_insights)

@app.post(
    "/esg/analyse",
    response_model=AnalyseResponse,
    tags=["ESG Analysis"],
)
async def esg_analyse(payload: ESGInput):
    """
    Main analysis endpoint used by Dashboard.jsx
    """
    global last_esg_input, last_scores, last_insights

    scores = calculate_esg_scores(payload)
    insights = await generate_esg_ai_insights(payload, scores)

    last_esg_input = payload
    last_scores = scores
    last_insights = insights

    # LIVE: push snapshot to any connected dashboards
    await push_live_ai_update()

    return AnalyseResponse(scores=scores, insights=insights)

@app.post(
    "/api/esg-upload",
    response_model=ESGDataResponse,
    tags=["AI Insights"],
)
async def api_esg_upload(file: UploadFile = File(...)):
    """
    Upload ESG data (JSON or Excel), update the current ESG run and
    return mockData + insights similar to /api/esg-data`.
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

def parse_invoice_pdf(content: bytes, filename: str) -> InvoiceSummary:
    """
    Parse invoice PDF and extract structured data with 6-month history.
    """
    # Mock implementation - in real app, would use PDF parsing library
    import random
    from datetime import datetime, timedelta
    
    # Generate mock data
    company_name = f"Utility Company {random.randint(1, 10)}"
    invoice_date = datetime.now().strftime("%Y-%m-%d")
    total_energy = random.uniform(1000, 10000)
    total_charges = random.uniform(5000, 50000)
    
    # Generate 6-month history
    history = []
    for i in range(6):
        month_label = (datetime.now() - timedelta(days=30 * (5 - i))).strftime("%b-%y")
        energy = total_energy * random.uniform(0.8, 1.2) / 6
        charges = total_charges * random.uniform(0.8, 1.2) / 6
        carbon = energy * 0.85 / 1000  # Convert to tons
        
        history.append(InvoiceMonthHistory(
            month_label=month_label,
            energyKWh=round(energy, 2),
            total_current_charges=round(charges, 2),
            carbonTco2e=round(carbon, 3)
        ))
    
    return InvoiceSummary(
        filename=filename,
        company_name=company_name,
        invoice_date=invoice_date,
        total_energy_kwh=round(total_energy, 2),
        total_current_charges=round(total_charges, 2),
        total_amount_due=round(total_charges * 1.15, 2),  # Add VAT
        sixMonthHistory=history
    )

@app.post(
    "/api/invoice-upload",
    response_model=InvoiceSummary,
    tags=["Invoices"],
)
async def api_invoice_upload(file: UploadFile = File(...)):
    """
    Upload a single invoice PDF and return a structured summary.
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

@app.get("/api/upload-history", tags=["Uploads"])
async def get_upload_history():
    """
    Get history of uploaded ESG and invoice files (simplified version).
    """
    history = []
    
    # Add last ESG upload info
    if last_esg_input:
        history.append({
            "type": "esg_data",
            "company": last_esg_input.company_name,
            "uploaded_at": datetime.now().isoformat(),
            "rows_count": len(last_esg_uploaded_rows)
        })
    
    # Add invoice uploads
    for invoice in last_invoice_summaries[:10]:  # Last 10
        history.append({
            "type": "invoice",
            "filename": invoice.filename,
            "company": invoice.company_name,
            "uploaded_at": datetime.now().isoformat(),
            "energy_kwh": invoice.total_energy_kwh
        })
    
    return {"uploads": history}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
