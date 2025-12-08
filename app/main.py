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
from dateutil.relativedelta import relativedelta

load_dotenv()

# ================== CONFIG ==================
# Storage paths - using persistent storage for uploaded data
DATA_DIR = os.getenv("ESG_DATA_DIR", "uploaded_data")
os.makedirs(DATA_DIR, exist_ok=True)

# Paths for different types of uploaded data
UPLOADED_FILES_DIR = os.path.join(DATA_DIR, "files")
UPLOADED_LOGO_DIR = os.path.join(DATA_DIR, "logos")
UPLOADED_ESG_DATA_PATH = os.path.join(DATA_DIR, "latest_esg_data.json")
UPLOADED_ESG_ROWS_PATH = os.path.join(DATA_DIR, "latest_esg_rows.json")
UPLOADED_INVOICES_PATH = os.path.join(DATA_DIR, "latest_invoices.json")

# Create directories if they don't exist
os.makedirs(UPLOADED_FILES_DIR, exist_ok=True)
os.makedirs(UPLOADED_LOGO_DIR, exist_ok=True)

# Carbon calculation constants
CARBON_EMISSION_FACTOR = 0.99 / 1000  # tCO2e per kWh
CARBON_FORMULA = "Estimated Carbon (tCO₂e) = Total Energy (kWh) × 0.99 / 1000"

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
    uploaded_date: Optional[datetime] = None
    
    @validator('uploaded_date', pre=True, always=True)
    def set_uploaded_date(cls, v):
        return v or datetime.now()

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
    uploaded_date: Optional[datetime] = None
    
    @validator('uploaded_date', pre=True, always=True)
    def set_uploaded_date(cls, v):
        return v or datetime.now()

class InvoiceMonthHistory(BaseModel):
    month_label: Optional[str] = None
    month_date: Optional[datetime] = None
    energyKWh: Optional[float] = None
    total_current_charges: Optional[float] = None
    total_amount_due: Optional[float] = None
    maximum_demand_kva: Optional[float] = None
    carbonTco2e: Optional[float] = None
    calculated_at: Optional[datetime] = None
    
    @validator('carbonTco2e', pre=True, always=True)
    def calculate_carbon(cls, v, values):
        """Calculate carbon using formula: energy * 0.99 / 1000"""
        if v is not None:
            return v
        
        energy = values.get('energyKWh')
        if energy is not None and energy > 0:
            return round(energy * CARBON_EMISSION_FACTOR, 3)
        return None
    
    @validator('calculated_at', pre=True, always=True)
    def set_calculated_at(cls, v):
        return v or datetime.now()
    
    @validator('month_date', pre=True, always=True)
    def parse_month_date(cls, v, values):
        """Parse month_label to datetime if month_date not provided"""
        if v:
            return v
        
        month_label = values.get('month_label')
        if month_label:
            try:
                # Try parsing formats like "Oct-24", "October-2024"
                for fmt in ["%b-%y", "%B-%Y", "%b %y", "%B %Y", "%m-%y", "%m/%Y"]:
                    try:
                        return datetime.strptime(month_label, fmt)
                    except ValueError:
                        continue
            except:
                pass
        return None

class InvoiceSummary(BaseModel):
    filename: str
    company_name: str
    invoice_date: str
    uploaded_date: Optional[datetime] = None
    parsed_at: Optional[datetime] = None
    total_energy_kwh: Optional[float] = None
    total_current_charges: Optional[float] = None
    total_amount_due: Optional[float] = None
    sixMonthHistory: List[InvoiceMonthHistory] = []
    
    @validator('uploaded_date', pre=True, always=True)
    def set_uploaded_date(cls, v):
        return v or datetime.now()
    
    @validator('parsed_at', pre=True, always=True)
    def set_parsed_at(cls, v):
        return v or datetime.now()
    
    @validator('sixMonthHistory', pre=True)
    def sort_and_calculate_six_month(cls, v):
        """Sort history by date and ensure we have exactly 6 months"""
        if not v:
            return []
        
        # Sort by date
        sorted_history = sorted(
            v,
            key=lambda x: x.get('month_date') or _parse_month_label(x.get('month_label')) or datetime.min
        )
        
        # Take latest 6 months
        return sorted_history[-6:]
    
    @property
    def six_month_total_energy(self) -> float:
        """Calculate total energy for the 6-month period"""
        return sum(month.energyKWh or 0 for month in self.sixMonthHistory)
    
    @property
    def six_month_total_charges(self) -> float:
        """Calculate total charges for the 6-month period"""
        return sum(month.total_current_charges or 0 for month in self.sixMonthHistory)
    
    @property
    def six_month_total_carbon(self) -> float:
        """Calculate total carbon for the 6-month period using formula"""
        total_energy = self.six_month_total_energy
        return round(total_energy * CARBON_EMISSION_FACTOR, 3)
    
    @property
    def average_monthly_energy(self) -> float:
        """Calculate average monthly energy"""
        if not self.sixMonthHistory:
            return 0.0
        return self.six_month_total_energy / len(self.sixMonthHistory)
    
    @property
    def average_monthly_charges(self) -> float:
        """Calculate average monthly charges"""
        if not self.sixMonthHistory:
            return 0.0
        return self.six_month_total_charges / len(self.sixMonthHistory)
    
    @property
    def average_monthly_carbon(self) -> float:
        """Calculate average monthly carbon"""
        if not self.sixMonthHistory:
            return 0.0
        return self.six_month_total_carbon / len(self.sixMonthHistory)

class WebSocketMessage(BaseModel):
    type: str
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.now()

class SixMonthSummary(BaseModel):
    total_invoices_analyzed: int
    six_month_energy_kwh: float
    six_month_charges_r: float
    six_month_carbon_tonnes: float
    average_monthly_energy_kwh: float
    average_monthly_charges_r: float
    average_monthly_carbon_tonnes: float
    carbon_factor_used: float = CARBON_EMISSION_FACTOR
    carbon_formula: str = CARBON_FORMULA
    calculation_date: Optional[datetime] = None
    
    @validator('calculation_date', pre=True, always=True)
    def set_calculation_date(cls, v):
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

# ================== STORAGE HELPERS ==================
def save_uploaded_file(file_content: bytes, filename: str, category: str = "general") -> Dict[str, Any]:
    """
    Save uploaded file to persistent storage.
    Returns metadata including uploaded date.
    """
    # Sanitize filename
    safe_filename = re.sub(r'[^\w\.\-]', '_', filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{safe_filename}"
    
    file_path = os.path.join(UPLOADED_FILES_DIR, unique_filename)
    
    with open(file_path, 'wb') as f:
        f.write(file_content)
    
    # Create metadata with uploaded date
    uploaded_date = datetime.now()
    metadata = {
        "original_filename": filename,
        "safe_filename": safe_filename,
        "unique_filename": unique_filename,
        "category": category,
        "uploaded_at": uploaded_date.isoformat(),
        "uploaded_date": uploaded_date.isoformat(),
        "file_size": len(file_content),
        "file_path": file_path,
        "file_url": f"/api/storage/file/{unique_filename}"
    }
    
    metadata_path = file_path + ".meta.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return metadata

def get_latest_uploaded_files(category: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get latest uploaded files with metadata including uploaded date.
    """
    files = []
    
    for filename in os.listdir(UPLOADED_FILES_DIR):
        if filename.endswith('.meta.json'):
            metadata_path = os.path.join(UPLOADED_FILES_DIR, filename)
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                if category and metadata.get('category') != category:
                    continue
                    
                # Add file content if available
                file_path = metadata.get('file_path')
                if file_path and os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        metadata['file_content_base64'] = base64.b64encode(f.read()).decode('utf-8')
                
                # Ensure uploaded_date is properly set
                if 'uploaded_date' not in metadata and 'uploaded_at' in metadata:
                    metadata['uploaded_date'] = metadata['uploaded_at']
                
                files.append(metadata)
            except Exception as e:
                print(f"Error loading metadata {filename}: {e}")
    
    # Sort by upload time, newest first
    files.sort(key=lambda x: x.get('uploaded_date', ''), reverse=True)
    return files[:limit]

def save_esg_data(esg_data: Dict[str, Any]) -> str:
    """
    Save ESG data to persistent storage with timestamp.
    """
    if 'saved_at' not in esg_data:
        esg_data['saved_at'] = datetime.now().isoformat()
    esg_data['version'] = esg_data.get('version', '1.0')
    
    with open(UPLOADED_ESG_DATA_PATH, 'w') as f:
        json.dump(esg_data, f, indent=2, default=str)
    
    return UPLOADED_ESG_DATA_PATH

def load_latest_esg_data() -> Optional[Dict[str, Any]]:
    """
    Load latest ESG data from storage.
    """
    if os.path.exists(UPLOADED_ESG_DATA_PATH):
        try:
            with open(UPLOADED_ESG_DATA_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading ESG data: {e}")
    return None

def save_esg_rows(rows: List[Dict[str, Any]]) -> str:
    """
    Save ESG rows to persistent storage with timestamp.
    """
    data = {
        'rows': rows,
        'saved_at': datetime.now().isoformat(),
        'row_count': len(rows),
        'uploaded_date': datetime.now().isoformat()
    }
    
    with open(UPLOADED_ESG_ROWS_PATH, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    return UPLOADED_ESG_ROWS_PATH

def load_latest_esg_rows() -> List[Dict[str, Any]]:
    """
    Load latest ESG rows from storage.
    """
    if os.path.exists(UPLOADED_ESG_ROWS_PATH):
        try:
            with open(UPLOADED_ESG_ROWS_PATH, 'r') as f:
                data = json.load(f)
                return data.get('rows', [])
        except Exception as e:
            print(f"Error loading ESG rows: {e}")
    return []

def save_invoices(invoices: List[Dict[str, Any]]) -> str:
    """
    Save invoices to persistent storage with timestamp.
    """
    data = {
        'invoices': invoices,
        'saved_at': datetime.now().isoformat(),
        'invoice_count': len(invoices),
        'uploaded_date': datetime.now().isoformat()
    }
    
    with open(UPLOADED_INVOICES_PATH, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    return UPLOADED_INVOICES_PATH

def load_latest_invoices() -> List[Dict[str, Any]]:
    """
    Load latest invoices from storage.
    """
    if os.path.exists(UPLOADED_INVOICES_PATH):
        try:
            with open(UPLOADED_INVOICES_PATH, 'r') as f:
                data = json.load(f)
                return data.get('invoices', [])
        except Exception as e:
            print(f"Error loading invoices: {e}")
    return []

def save_logo(logo_content: bytes, source: str = "upload") -> Dict[str, Any]:
    """
    Save logo to persistent storage with timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logo_filename = f"logo_{timestamp}_{source}.png"
    logo_path = os.path.join(UPLOADED_LOGO_DIR, logo_filename)
    
    with open(logo_path, 'wb') as f:
        f.write(logo_content)
    
    # Save metadata with uploaded date
    uploaded_date = datetime.now()
    metadata = {
        "filename": logo_filename,
        "source": source,
        "saved_at": uploaded_date.isoformat(),
        "uploaded_date": uploaded_date.isoformat(),
        "file_size": len(logo_content),
        "file_path": logo_path
    }
    
    metadata_path = logo_path + ".meta.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def get_latest_logo() -> Optional[Dict[str, Any]]:
    """
    Get latest uploaded logo with metadata including uploaded date.
    """
    logos = []
    
    for filename in os.listdir(UPLOADED_LOGO_DIR):
        if filename.endswith('.meta.json'):
            metadata_path = os.path.join(UPLOADED_LOGO_DIR, filename)
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logos.append(metadata)
            except Exception as e:
                print(f"Error loading logo metadata {filename}: {e}")
    
    if logos:
        # Sort by uploaded date, newest first
        logos.sort(key=lambda x: x.get('uploaded_date', x.get('saved_at', '')), reverse=True)
        latest_logo = logos[0]
        
        # Add logo content
        logo_path = latest_logo.get('file_path')
        if logo_path and os.path.exists(logo_path):
            with open(logo_path, 'rb') as f:
                logo_content = f.read()
                latest_logo['logo_content_base64'] = base64.b64encode(logo_content).decode('utf-8')
                latest_logo['logo_content'] = logo_content
        
        return latest_logo
    
    return None

# ================== MODIFY EXISTING FUNCTIONS TO USE STORAGE ==================

def load_last_esg_from_disk() -> ESGInput:
    """
    Load last_esg_input from storage if available; fall back to DEFAULT_ESG_INPUT.
    """
    stored_data = load_latest_esg_data()
    if stored_data and 'input' in stored_data:
        try:
            input_data = stored_data['input']
            # Ensure uploaded_date is set
            if 'uploaded_date' not in input_data:
                input_data['uploaded_date'] = stored_data.get('saved_at', datetime.now().isoformat())
            return ESGInput(**input_data)
        except Exception as exc:
            print(f"Failed to load ESG input from storage: {exc}")
    
    return DEFAULT_ESG_INPUT

def save_last_esg_to_disk(esg_input: ESGInput) -> None:
    """
    Persist last_esg_input to storage with timestamp.
    """
    try:
        stored_data = load_latest_esg_data() or {}
        stored_data['input'] = esg_input.dict()
        stored_data['last_updated'] = datetime.now().isoformat()
        stored_data['uploaded_date'] = esg_input.uploaded_date.isoformat() if esg_input.uploaded_date else datetime.now().isoformat()
        save_esg_data(stored_data)
    except Exception as exc:
        print(f"Failed to save ESG input to storage: {exc}")

def load_last_esg_rows_from_disk() -> List[Dict[str, Any]]:
    """
    Load the last uploaded ESG rows from storage.
    """
    rows = load_latest_esg_rows()
    # Add uploaded_date if not present
    for row in rows:
        if 'uploaded_date' not in row:
            row['uploaded_date'] = datetime.now().isoformat()
    return rows

def save_last_esg_rows_to_disk(rows: List[Dict[str, Any]]) -> None:
    """
    Persist latest ESG raw rows to storage with timestamp.
    """
    # Add uploaded_date to each row
    uploaded_date = datetime.now()
    for row in rows:
        if 'uploaded_date' not in row:
            row['uploaded_date'] = uploaded_date.isoformat()
    save_esg_rows(rows)

def load_last_invoices_from_disk() -> List[InvoiceSummary]:
    """
    Load previously parsed invoice summaries from storage.
    """
    stored_invoices = load_latest_invoices()
    invoices: List[InvoiceSummary] = []
    
    for item in stored_invoices:
        try:
            # Ensure uploaded_date is set
            if 'uploaded_date' not in item:
                item['uploaded_date'] = datetime.now().isoformat()
            invoices.append(InvoiceSummary(**item))
        except Exception:
            # Skip any bad record rather than failing the whole load
            continue
    
    return invoices

def save_last_invoices_to_disk(invoices: List[InvoiceSummary]) -> None:
    """
    Persist invoice summaries to storage with timestamp.
    """
    serializable = [inv.dict() for inv in invoices]
    save_invoices(serializable)

# ================== ENHANCED INVOICE AGGREGATION ==================

def aggregate_six_month_charges(invoices: List[InvoiceSummary]) -> Dict[str, Any]:
    """
    Calculate total charges for the latest 6 months across all invoices.
    This aggregates data from the sixMonthHistory of each invoice.
    """
    if not invoices:
        return {
            "total_energy_kwh": 0.0,
            "total_charges_r": 0.0,
            "total_carbon_tco2e": 0.0,
            "average_monthly_energy_kwh": 0.0,
            "average_monthly_charges_r": 0.0,
            "average_monthly_carbon_tco2e": 0.0,
            "monthly_breakdown": [],
            "invoice_count": 0,
            "calculation_date": datetime.now().isoformat(),
            "carbon_formula": CARBON_FORMULA
        }
    
    # Collect all monthly data across all invoices
    monthly_data: Dict[str, Dict[str, float]] = {}  # month_key -> {energy, charges, carbon}
    
    for invoice in invoices:
        if invoice.sixMonthHistory:
            for month in invoice.sixMonthHistory:
                month_key = month.month_label or month.month_date.strftime("%b-%Y") if month.month_date else "Unknown"
                
                if month_key not in monthly_data:
                    monthly_data[month_key] = {
                        "month": month_key,
                        "month_date": month.month_date.isoformat() if month.month_date else None,
                        "energy_kwh": 0.0,
                        "charges_r": 0.0,
                        "carbon_tco2e": 0.0
                    }
                
                monthly_data[month_key]["energy_kwh"] += month.energyKWh or 0.0
                monthly_data[month_key]["charges_r"] += month.total_current_charges or 0.0
                
                # Calculate carbon for this month's energy
                month_energy = month.energyKWh or 0.0
                month_carbon = month_energy * CARBON_EMISSION_FACTOR
                monthly_data[month_key]["carbon_tco2e"] += month_carbon
    
    # Convert to list and sort by date
    monthly_list = list(monthly_data.values())
    
    # Sort by date
    def sort_key(item):
        date_str = item.get("month_date") or item.get("month", "")
        try:
            # Try to parse various date formats
            for fmt in ["%Y-%m-%dT%H:%M:%S", "%b-%Y", "%B-%Y", "%m-%Y"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except:
                    continue
            return datetime.min
        except:
            return datetime.min
    
    monthly_list.sort(key=sort_key)
    
    # Take only the latest 6 months
    latest_six_months = monthly_list[-6:] if len(monthly_list) > 6 else monthly_list
    
    # Calculate totals
    total_energy = sum(item["energy_kwh"] for item in latest_six_months)
    total_charges = sum(item["charges_r"] for item in latest_six_months)
    total_carbon = sum(item["carbon_tco2e"] for item in latest_six_months)
    
    # Calculate averages
    month_count = len(latest_six_months) if latest_six_months else 1
    avg_monthly_energy = total_energy / month_count
    avg_monthly_charges = total_charges / month_count
    avg_monthly_carbon = total_carbon / month_count
    
    return {
        "total_energy_kwh": round(total_energy, 2),
        "total_charges_r": round(total_charges, 2),
        "total_carbon_tco2e": round(total_carbon, 3),
        "average_monthly_energy_kwh": round(avg_monthly_energy, 2),
        "average_monthly_charges_r": round(avg_monthly_charges, 2),
        "average_monthly_carbon_tco2e": round(avg_monthly_carbon, 3),
        "monthly_breakdown": latest_six_months,
        "invoice_count": len(invoices),
        "calculation_date": datetime.now().isoformat(),
        "carbon_formula": CARBON_FORMULA,
        "carbon_factor": CARBON_EMISSION_FACTOR
    }

def calculate_six_month_totals_from_invoices(invoices: List[InvoiceSummary]) -> Tuple[float, float, float]:
    """
    Calculate total energy, charges, and carbon for the latest 6 months.
    
    Returns:
        Tuple of (total_energy_kwh, total_charges_r, total_carbon_tco2e)
    """
    aggregated = aggregate_six_month_charges(invoices)
    
    return (
        aggregated["total_energy_kwh"],
        aggregated["total_charges_r"],
        aggregated["total_carbon_tco2e"]
    )

# ================== UPDATE BUILD_SUMMARY_AND_METRICS ==================

def build_summary_and_metrics(esg_input: ESGInput, scores: ESGScores):
    """
    Build summary + metrics blocks used by the dashboard.
    
    Uses correct carbon formula: Estimated Carbon = Total Energy * 0.99 / 1000
    Includes 6-month totals from invoices.
    """
    from math import isnan

    global last_esg_uploaded_rows, last_invoice_summaries

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

    # ---------- Calculate 6-month totals from invoices ----------
    six_month_energy = 0.0
    six_month_charges = 0.0
    six_month_carbon = 0.0
    
    if last_invoice_summaries:
        try:
            six_month_energy, six_month_charges, six_month_carbon = calculate_six_month_totals_from_invoices(last_invoice_summaries)
        except Exception as exc:
            print("Failed to calculate 6-month totals from invoices:", exc)
    
    # ---------- Carbon calculations ----------
    # Calculate carbon using correct formula
    carbon_for_metrics = total_energy * CARBON_EMISSION_FACTOR
    
    # If we have invoice data, use that carbon instead
    if six_month_carbon > 0:
        carbon_for_metrics = six_month_carbon

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
            "totalEnergyConsumption": total_energy,  # kWh (invoice-preferred)
            "sixMonthEnergyConsumption": six_month_energy,  # kWh (latest 6 months)
            "totalFuelUsageLitres": total_fuel,  # L
            "renewableEnergyShare": renewable_share,  # %
            "carbonEmissions": round(carbon_for_metrics, 3),  # tCO₂e (invoice-preferred)
            "sixMonthCarbonEmissions": round(six_month_carbon, 3),  # tCO₂e (latest 6 months)
            "carbonFormula": CARBON_FORMULA,
            "lastCalculated": datetime.now().isoformat()
        },
        "social": {
            "supplierDiversity": soc_metrics["supplierDiversity"],
            "customerSatisfaction": scores.s_score,
            "humanCapital": round((scores.s_score + scores.overall_score) / 2),
            "lastUpdated": datetime.now().isoformat()
        },
        "governance": {
            "corporateGovernance": gov_metrics["corporateGovernance"],
            "iso9001Compliance": gov_metrics["isoCompliance"],
            "businessEthics": gov_metrics["businessEthics"],
            "totalGovernanceTrainings": gov_metrics["totalGovernanceTrainings"],
            "totalEnvironmentalTrainings": gov_metrics["totalEnvironmentalTrainings"],
            "totalComplianceFindings": gov_metrics["totalComplianceFindings"],
            "lastUpdated": datetime.now().isoformat()
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
        # Six Month Carbon Tax (R) = six_month_carbon × tax rate
        "sixMonthCarbonTax": round(six_month_carbon * CARBON_TAX_RATE),
        # Total Charges (R) - from 6-month invoice aggregation
        "totalCharges": round(six_month_charges),
        # Six Month Charges (R)
        "sixMonthCharges": round(six_month_charges),
        # Applicable Tax Allowances (R) – assumed 30% of carbon tax
        "taxAllowances": round(
            carbon_for_metrics * CARBON_TAX_RATE * TAX_ALLOWANCE_FACTOR
        ),
        # Carbon Credits Generated (t) – assumed 15% of emissions as a proxy
        "carbonCredits": round(carbon_for_metrics * CARBON_CREDIT_FACTOR),
        # Energy Savings (kWh) – assumed 12% of total energy as a proxy
        "energySavings": round(total_energy * ENERGY_SAVINGS_FACTOR),
        # Six Month Energy (kWh)
        "sixMonthEnergy": round(six_month_energy),
        # Carbon Formula
        "carbonFormula": CARBON_FORMULA,
        # Timestamps
        "calculatedAt": datetime.now().isoformat(),
        "dataSource": "ESG Input + Invoice Aggregation"
    }

    return summary, metrics, env_metrics, soc_metrics, gov_metrics

# ================== UPDATE INVOICE PARSING ==================

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
    
    # Generate 6-month history (latest 6 months)
    history = []
    for i in range(6):
        # Calculate month date (going backwards from current month)
        month_date = datetime.now() - relativedelta(months=i)
        month_label = month_date.strftime("%b-%y")
        
        # Generate data for this month
        energy = total_energy * random.uniform(0.8, 1.2) / 6
        charges = total_charges * random.uniform(0.8, 1.2) / 6
        
        # Calculate carbon using correct formula: Energy * 0.99 / 1000
        carbon = energy * CARBON_EMISSION_FACTOR
        
        history.append(InvoiceMonthHistory(
            month_label=month_label,
            month_date=month_date,
            energyKWh=round(energy, 2),
            total_current_charges=round(charges, 2),
            carbonTco2e=round(carbon, 3),
            calculated_at=datetime.now()
        ))
    
    # Reverse to have oldest first
    history.reverse()
    
    return InvoiceSummary(
        filename=filename,
        company_name=company_name,
        invoice_date=invoice_date,
        uploaded_date=datetime.now(),
        total_energy_kwh=round(total_energy, 2),
        total_current_charges=round(total_charges, 2),
        total_amount_due=round(total_charges * 1.15, 2),  # Add VAT
        sixMonthHistory=history
    )

# ================== UPDATE OTHER FUNCTIONS ==================

def build_environmental_metrics_from_input(
    esg_input: ESGInput, scores: ESGScores
) -> Dict[str, Any]:
    """
    Build time-series data for charts, preferring the raw uploaded ESG rows.
    
    Uses correct carbon formula: Estimated Carbon = Total Energy * 0.99 / 1000
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

            # CO₂ (t): robust detection from columns, else compute from energy using CORRECT formula
            co2_col = None

            # 1) Exact candidates first
            exact_carbon_cols = ["CO2 (t)", "Carbon Emissions (t)", "Emissions (tCO2e)"]
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
                    df[co2_col].fillna(0).apply(_safe_float).round(3).tolist()
                )
            else:
                # 3) Fallback: compute from energy using CORRECT formula
                if energy_col:
                    energy_series = (
                        df[energy_col].fillna(0).apply(_safe_float)
                        if energy_col
                        else pd.Series([0.0] * len(df))
                    )
                    co2_emissions = (
                        energy_series * CARBON_EMISSION_FACTOR
                    ).round(3).tolist()

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
                    "calculated_at": datetime.now().isoformat()
                }

        except Exception as exc:
            # If anything fails here, don't crash – just log and fall back
            print("Failed to build environmental metrics from uploaded rows:", exc)

    # ---------- 2) FALLBACK: synthetic series from aggregated ESGInput ----------
    periods = 6
    base_energy_kwh = esg_input.energy_consumption_mwh * 1000
    base_co2 = base_energy_kwh * CARBON_EMISSION_FACTOR  # CORRECT: Use formula
    total_fuel_l = getattr(esg_input, "fuel_litres", 0.0) or 0.0

    energy_usage: List[float] = []
    co2_emissions: List[float] = []
    waste: List[float] = []
    fuel_usage: List[float] = []

    for i in range(periods):
        factor = 0.8 + 0.05 * i
        energy_usage.append(round(base_energy_kwh / periods * factor))
        co2_emissions.append(round(base_co2 / periods * factor, 3))  # More precise
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
        "calculated_at": datetime.now().isoformat()
    }

# ================== ADD NEW ROUTE FOR 6-MONTH ANALYSIS ==================

@app.get("/api/analytics/six-month-totals", response_model=SixMonthSummary, tags=["Analytics"])
async def get_six_month_totals():
    """
    Get comprehensive 6-month totals from all invoices.
    Calculates total energy, charges, and carbon using formula: carbon = energy * 0.99 / 1000
    """
    global last_invoice_summaries
    
    if not last_invoice_summaries:
        raise HTTPException(
            status_code=404,
            detail="No invoices available for analysis. Please upload invoices first."
        )
    
    aggregated = aggregate_six_month_charges(last_invoice_summaries)
    
    summary = SixMonthSummary(
        total_invoices_analyzed=aggregated["invoice_count"],
        six_month_energy_kwh=aggregated["total_energy_kwh"],
        six_month_charges_r=aggregated["total_charges_r"],
        six_month_carbon_tonnes=aggregated["total_carbon_tco2e"],
        average_monthly_energy_kwh=aggregated["average_monthly_energy_kwh"],
        average_monthly_charges_r=aggregated["average_monthly_charges_r"],
        average_monthly_carbon_tonnes=aggregated["average_monthly_carbon_tco2e"],
        calculation_date=datetime.fromisoformat(aggregated["calculation_date"])
    )
    
    return summary

@app.get("/api/analytics/invoice-breakdown", tags=["Analytics"])
async def get_invoice_breakdown():
    """
    Get detailed breakdown of each invoice's 6-month data.
    """
    global last_invoice_summaries
    
    if not last_invoice_summaries:
        raise HTTPException(
            status_code=404,
            detail="No invoices available for analysis."
        )
    
    breakdown = []
    for invoice in last_invoice_summaries:
        breakdown.append({
            "filename": invoice.filename,
            "company_name": invoice.company_name,
            "invoice_date": invoice.invoice_date,
            "uploaded_date": invoice.uploaded_date.isoformat() if invoice.uploaded_date else None,
            "total_energy_kwh": invoice.total_energy_kwh,
            "total_current_charges": invoice.total_current_charges,
            "six_month_energy_kwh": invoice.six_month_total_energy,
            "six_month_charges_r": invoice.six_month_total_charges,
            "six_month_carbon_tco2e": invoice.six_month_total_carbon,
            "average_monthly_energy_kwh": invoice.average_monthly_energy,
            "average_monthly_charges_r": invoice.average_monthly_charges,
            "average_monthly_carbon_tco2e": invoice.average_monthly_carbon,
            "month_count": len(invoice.sixMonthHistory),
            "carbon_formula": CARBON_FORMULA
        })
    
    # Also include aggregated totals
    aggregated = aggregate_six_month_charges(last_invoice_summaries)
    
    return {
        "invoice_breakdown": breakdown,
        "aggregated_totals": aggregated,
        "total_invoices": len(last_invoice_summaries),
        "analysis_date": datetime.now().isoformat()
    }

# ================== UPDATE EXISTING ROUTES ==================

@app.post(
    "/api/esg-upload",
    response_model=ESGDataResponse,
    tags=["AI Insights"],
)
async def api_esg_upload(file: UploadFile = File(...)):
    """
    Upload ESG data (JSON or Excel), update the current ESG run and
    return mockData + insights similar to /api/esg-data`.
    
    Now saves the uploaded file to persistent storage with uploaded date.
    """
    global last_esg_input, last_scores, last_insights, last_esg_uploaded_rows

    filename = (file.filename or "").lower()
    uploaded_date = datetime.now()

    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {exc}",
        )
    
    # Save the uploaded file to storage
    metadata = save_uploaded_file(content, filename, "esg_data")

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
        
        # Add uploaded date
        clean_payload["uploaded_date"] = uploaded_date

        try:
            esg_input = ESGInput(**clean_payload)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"JSON does not match ESGInput schema: {exc}",
            )

    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        esg_input, uploaded_rows = build_esg_input_from_excel(content)
        # Add uploaded date to ESG input
        esg_input.uploaded_date = uploaded_date
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

    # Add uploaded date to each row
    for row in last_esg_uploaded_rows:
        row["uploaded_date"] = uploaded_date.isoformat()

    # Save to persistent storage
    save_last_esg_to_disk(last_esg_input)
    save_last_esg_rows_to_disk(last_esg_uploaded_rows)
    
    # Save complete upload result
    upload_result = {
        "filename": filename,
        "file_path": metadata.get("file_path"),
        "metadata": metadata,
        "input": esg_input.dict(),
        "scores": scores.dict(),
        "insights": insights_obj.dict(),
        "uploaded_rows_count": len(uploaded_rows),
        "uploaded_date": uploaded_date.isoformat(),
        "uploaded_at": uploaded_date.isoformat()
    }
    save_esg_data(upload_result)

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
        last_updated=uploaded_date
    )

    # LIVE: ESG dataset changed -> push snapshot
    await push_live_ai_update()

    return ESGDataResponse(
        mockData=mock,
        insights=combined_insights,
        uploaded_date=uploaded_date
    )

@app.post(
    "/api/invoice-upload",
    response_model=InvoiceSummary,
    tags=["Invoices"],
)
async def api_invoice_upload(file: UploadFile = File(...)):
    """
    Upload a single invoice PDF and return a structured summary.
    Now saves the uploaded file to persistent storage with uploaded date.
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
    
    # Save the uploaded file to storage
    metadata = save_uploaded_file(content, filename, "invoice")

    summary = parse_invoice_pdf(content, filename)
    
    # Ensure uploaded_date is set
    summary.uploaded_date = datetime.now()

    # store for dashboard (latest first)
    last_invoice_summaries.insert(0, summary)
    if len(last_invoice_summaries) > 200:
        last_invoice_summaries.pop()

    # Persist updated invoices to storage
    save_last_invoices_to_disk(last_invoice_summaries)

    # LIVE: invoice changed -> push update
    await push_live_ai_update()

    return summary

# ================== GLOBAL STATE INITIALIZATION ==================
last_esg_input: ESGInput = load_last_esg_from_disk()
last_scores: Optional[ESGScores] = None
last_insights: Optional[ESGInsights] = None
last_esg_uploaded_rows: List[Dict[str, Any]] = load_last_esg_rows_from_disk()
last_invoice_summaries: List[InvoiceSummary] = load_last_invoices_from_disk()
last_extracted_logo: Optional[str] = None

# Load latest logo from storage
latest_logo = get_latest_logo()
if latest_logo:
    last_extracted_logo = latest_logo.get('logo_content_base64')

print(f"Initialized with {len(last_esg_uploaded_rows)} ESG rows and {len(last_invoice_summaries)} invoices from storage")
print(f"Carbon calculation formula: {CARBON_FORMULA}")

# ================== HELPER FUNCTIONS ==================
def _parse_month_label(label: str) -> Optional[datetime]:
    """Parse month label like 'Oct-24' to datetime."""
    try:
        return datetime.strptime(label, "%b-%y")
    except ValueError:
        try:
            return datetime.strptime(label, "%B-%y")
        except ValueError:
            return None

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
            uploaded_date=datetime.now()
        )
        
        return esg_input, rows
    except Exception as e:
        print(f"Error parsing Excel: {e}")
        # Return default if parsing fails
        return ESGInput(uploaded_date=datetime.now()), []

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
    overall = f"Company {esg_input.company_name} shows {'strong' if scores.overall_score > 70 else 'moderate' if scores.overall_score > 50 else 'needs improvement'} ESG performance with overall score of {scores.overall_score}/100. Analysis completed on {datetime.now().strftime('%Y-%m-%d')}."
    
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
    global last_esg_input, last_scores, last_insights, last_invoice_summaries
    
    if not last_esg_input or not last_scores:
        return
    
    # Calculate 6-month totals
    six_month_totals = {}
    if last_invoice_summaries:
        six_month_totals = aggregate_six_month_charges(last_invoice_summaries)
    
    snapshot = {
        "type": "esg_snapshot",
        "timestamp": datetime.now().isoformat(),
        "uploaded_date": last_esg_input.uploaded_date.isoformat() if last_esg_input.uploaded_date else datetime.now().isoformat(),
        "input": last_esg_input.dict() if hasattr(last_esg_input, 'dict') else {},
        "scores": last_scores.dict() if last_scores else {},
        "six_month_totals": six_month_totals,
        "invoice_count": len(last_invoice_summaries),
        "carbon_formula": CARBON_FORMULA
    }
    
    message = WebSocketMessage(type="esg_update", data=snapshot)
    
    for connection in active_connections:
        try:
            await connection.send_json(message.dict())
        except Exception as e:
            print(f"Failed to send WebSocket update: {e}")

# ================== ADDITIONAL ROUTES ==================
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "ESG Analysis API", 
        "version": "1.0.0",
        "carbon_formula": CARBON_FORMULA,
        "uploaded_date": datetime.now().isoformat()
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
        last_updated=last_esg_input.uploaded_date if last_esg_input.uploaded_date else datetime.now()
    )
    
    combined_insights: List[str] = []
    if last_insights:
        combined_insights = [
            last_insights.overall,
            *last_insights.environmental,
            *last_insights.social,
            *last_insights.governance,
        ]
    
    return ESGDataResponse(
        mockData=mock, 
        insights=combined_insights,
        uploaded_date=last_esg_input.uploaded_date if last_esg_input.uploaded_date else datetime.now()
    )

@app.post(
    "/esg/analyse",
    response_model=AnalyseResponse,
    tags=["ESG Analysis"],
)
async def esg_analyse(payload: ESGInput):
    """
    Main analysis endpoint used by Dashboard.jsx.
    Now saves the ESG input to storage.
    """
    global last_esg_input, last_scores, last_insights

    # Ensure uploaded_date is set
    if not payload.uploaded_date:
        payload.uploaded_date = datetime.now()

    scores = calculate_esg_scores(payload)
    insights = await generate_esg_ai_insights(payload, scores)

    last_esg_input = payload
    last_scores = scores
    last_insights = insights

    # Save to persistent storage
    save_last_esg_to_disk(last_esg_input)
    
    # Also save the complete analysis result
    analysis_result = {
        "input": payload.dict(),
        "scores": scores.dict(),
        "insights": insights.dict(),
        "analysed_at": datetime.now().isoformat(),
        "uploaded_date": payload.uploaded_date.isoformat()
    }
    save_esg_data(analysis_result)

    # LIVE: push snapshot to any connected dashboards
    await push_live_ai_update()

    return AnalyseResponse(scores=scores, insights=insights)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)