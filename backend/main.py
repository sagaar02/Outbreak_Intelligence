import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import os

# --- PATH CONFIGURATION ---
# Resolve the project root (one level above backend/) using the location of this file.
# This ensures paths work correctly regardless of where uvicorn is launched from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- CONFIGURATION & CONSTANTS ---
OUTBREAK_THRESHOLD = 0.35
RISK_LEVEL_HIGH = 0.38
RISK_LEVEL_MEDIUM = 0.20

# Model & Data paths resolved from the project root
MODEL_PATH = os.path.join(BASE_DIR, "model", "outbreak_model.pkl")
DATA_FOLDER = os.path.join(BASE_DIR, "DATA")
DASHBOARD_PATH = os.path.join(BASE_DIR, "dashboard.html")

# Initialize FastAPI application
app = FastAPI(
    title="AI Outbreak Prediction System (Recall-Optimized)",
    description="Advanced Backend API utilizing optimized thresholds for early disease detection.",
    version="1.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
model = None

@app.on_event("startup")
def load_production_model():
    """Securely load the trained ML pipeline on application startup."""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Production model loaded from {MODEL_PATH}")
        else:
            raise FileNotFoundError(f"Critical Error: Outbreak model not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Startup Failure: {str(e)}")

# --- DATA SCHEMA ---
class PredictionInput(BaseModel):
    """
    Pydantic schema for outbreak prediction.
    Fields are derived from health monitoring and environmental quality metrics.
    """
    diarrhea_cases: float = Field(..., description="Daily reported diarrhea cases")
    fever_cases: float = Field(..., description="Daily reported fever cases")
    vomiting_cases: float = Field(..., description="Daily reported vomiting cases")
    pH: float = Field(..., description="Water pH level")
    turbidity: float = Field(..., description="Water turbidity (NTU)")
    rainfall_mm: float = Field(..., description="Measured rainfall in mm")
    temperature: float = Field(..., description="Average ambient temperature (°C)")
    case_growth_rate: float = Field(..., description="Relative increase in cases compared to 7d moving average")

# --- ENDPOINTS ---

@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard HTML page."""
    if not os.path.exists(DASHBOARD_PATH):
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return FileResponse(DASHBOARD_PATH, media_type="text/html")

@app.get("/health")
def health_check():
    """System health pulse endpoint."""
    return {"status": "healthy", "model_active": model is not None}

@app.post("/predict")
async def predict_outbreak_risk(data: PredictionInput):
    """
    Predicts outbreak probability using a custom decision threshold for public health safety.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable. Inference engine initialization failed.")

    try:
        input_dict = data.dict()
        features_df = pd.DataFrame([input_dict])

        probabilities = model.predict_proba(features_df)
        outbreak_prob = float(probabilities[0][1])

        refined_prediction = 1 if outbreak_prob >= OUTBREAK_THRESHOLD else 0

        if outbreak_prob >= RISK_LEVEL_HIGH:
            risk_level = "HIGH"
        elif outbreak_prob >= RISK_LEVEL_MEDIUM:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "prediction": refined_prediction,
            "outbreak_probability": round(outbreak_prob, 4),
            "risk_level": risk_level,
            "metadata": {
                "decision_threshold_applied": OUTBREAK_THRESHOLD,
                "recall_prioritization": True,
                "model_version": "v1.1-research-grade"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failure: {str(e)}")

# --- ANALYTICS ENDPOINTS ---

def load_csv(filename):
    """Utility to load CSV from the DATA folder."""
    path = os.path.join(DATA_FOLDER, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@app.get("/api/v1/analytics/water-sources")
async def get_water_sources():
    """Returns distribution of water source types."""
    df = load_csv("water_pollution_disease.csv")
    if df is None:
        raise HTTPException(status_code=404, detail="Data file not found")
    
    counts = df['Water Source Type'].value_counts().to_dict()
    return {"labels": list(counts.keys()), "values": list(counts.values())}

@app.get("/api/v1/analytics/disease-stats")
async def get_disease_stats():
    """Returns aggregated disease cases per region."""
    df = load_csv("water_pollution_disease.csv")
    if df is None:
        raise HTTPException(status_code=404, detail="Data file not found")
    
    df.columns = [c.strip() for c in df.columns]
    
    region_stats = df.groupby('Region').agg({
        'Diarrheal Cases per 100,000 people': 'mean',
        'Cholera Cases per 100,000 people': 'mean',
        'Typhoid Cases per 100,000 people': 'mean'
    }).round(2).to_dict('index')
    
    return {
        "labels": list(region_stats.keys()),
        "datasets": [
            {"label": "Diarrheal", "data": [v['Diarrheal Cases per 100,000 people'] for v in region_stats.values()]},
            {"label": "Cholera", "data": [v['Cholera Cases per 100,000 people'] for v in region_stats.values()]},
            {"label": "Typhoid", "data": [v['Typhoid Cases per 100,000 people'] for v in region_stats.values()]}
        ]
    }

@app.get("/api/v1/analytics/rainfall-trends")
async def get_rainfall_trends():
    """Returns annual rainfall normal data for Indian states (Top 10)."""
    df = load_csv("district wise rainfall normal.csv")
    if df is None:
        raise HTTPException(status_code=404, detail="Data file not found")
    
    state_rainfall = df.groupby('STATE_UT_NAME')['ANNUAL'].mean().sort_values(ascending=False).head(15).round(2).to_dict()
    
    return {"labels": list(state_rainfall.keys()), "values": list(state_rainfall.values())}

@app.get("/api/v1/analytics/water-quality")
async def get_water_quality():
    """Returns pH and BOD ranges for Indian water bodies."""
    df = load_csv("Indian_water_data.csv")
    if df is None:
        raise HTTPException(status_code=404, detail="Data file not found")
    
    df['pH - Max'] = pd.to_numeric(df['pH - Max'], errors='coerce')
    df['BOD (mg/L) - Max'] = pd.to_numeric(df['BOD (mg/L) - Max'], errors='coerce')
    df = df.dropna(subset=['pH - Max', 'BOD (mg/L) - Max'])
    
    quality_stats = df.groupby('State Name').agg({
        'pH - Max': 'mean',
        'BOD (mg/L) - Max': 'mean'
    }).round(2).head(20).to_dict('index')
    
    return {
        "labels": list(quality_stats.keys()),
        "ph_values": [v['pH - Max'] for v in quality_stats.values()],
        "bod_values": [v['BOD (mg/L) - Max'] for v in quality_stats.values()]
    }
