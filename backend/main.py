import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import os

# --- CONFIGURATION & CONSTANTS ---
# 1. OUTBREAK_THRESHOLD: Decision boundary for binary classification.
# Set to 0.35 to prioritize RECALL (sensitivity). In public health, missing a potential 
# outbreak (False Negative) is significantly more costly than a false alarm.
OUTBREAK_THRESHOLD = 0.35

# 2. RISK_VISUALIZATION_THRESHOLDS: Qualitative mapping for UI feedback.
# These are calibrated to the model's actual score distribution (which plateaus near 0.42).
# This decoupling allows us to maintain a strict binary safety trigger while providing 
# granular nuance for "Warning" vs "Critical" states.
RISK_LEVEL_HIGH = 0.38
RISK_LEVEL_MEDIUM = 0.20

# Model Path Configuration
MODEL_PATH = os.path.join("..", "model", "outbreak_model.pkl")

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
        # Check primary path
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Production model loaded from {MODEL_PATH}")
        else:
            # Fallback path for different execution environments
            fallback = os.path.join("model", "outbreak_model.pkl")
            if os.path.exists(fallback):
                model = joblib.load(fallback)
                print(f"Fallback model loaded from {fallback}")
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
        # 1. Prepare data for the Scikit-Learn Pipeline
        # We maintain the feature order expected by the trained model
        input_dict = data.dict()
        features_df = pd.DataFrame([input_dict])

        # 2. Extract Positive Class Probability
        # Instead of model.predict() (default 0.5), we use probabilities to apply our custom sensitivity
        probabilities = model.predict_proba(features_df)
        outbreak_prob = float(probabilities[0][1])  # Index 1 corresponds to 'Outbreak = 1'

        # 3. Apply Decision Threshold (Recall Prioritization)
        # Prediction is 1 if probability exceeds OUTBREAK_THRESHOLD (0.35)
        refined_prediction = 1 if outbreak_prob >= OUTBREAK_THRESHOLD else 0

        # 4. Determine Qualitative Risk Level (Calibrated Mapping)
        # Note: We decouple visualization from the binary decision threshold.
        # This addresses model-specific probability compression while ensuring the 
        # 'Outbreak' flag remains sensitive to early indicators.
        if outbreak_prob >= RISK_LEVEL_HIGH:
            risk_level = "HIGH"
        elif outbreak_prob >= RISK_LEVEL_MEDIUM:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # 5. Return JSON payload
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

DATA_FOLDER = "DATA"

def load_csv(filename):
    """Utility to load CSV from the DATA folder."""
    path = os.path.join(DATA_FOLDER, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    # Try parent directory if running from backend folder
    path = os.path.join("..", DATA_FOLDER, filename)
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
    
    # Clean column names for easier access
    df.columns = [c.strip() for c in df.columns]
    
    # Group by Region and calculate mean cases
    # Using 'Diarrheal Cases per 100,000 people', 'Cholera Cases per 100,000 people', 'Typhoid Cases per 100,000 people'
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
    
    # Calculate average annual rainfall per state
    state_rainfall = df.groupby('STATE_UT_NAME')['ANNUAL'].mean().sort_values(ascending=False).head(15).round(2).to_dict()
    
    return {"labels": list(state_rainfall.keys()), "values": list(state_rainfall.values())}

@app.get("/api/v1/analytics/water-quality")
async def get_water_quality():
    """Returns pH and BOD ranges for Indian water bodies."""
    df = load_csv("Indian_water_data.csv")
    if df is None:
        raise HTTPException(status_code=404, detail="Data file not found")
    
    # Let's take a sample or aggregate
    # Filter out rows with non-numeric pH/BOD if any
    df['pH - Max'] = pd.to_numeric(df['pH - Max'], errors='coerce')
    df['BOD (mg/L) - Max'] = pd.to_numeric(df['BOD (mg/L) - Max'], errors='coerce')
    df = df.dropna(subset=['pH - Max', 'BOD (mg/L) - Max'])
    
    # Group by State and get max pH/BOD
    quality_stats = df.groupby('State Name').agg({
        'pH - Max': 'mean',
        'BOD (mg/L) - Max': 'mean'
    }).round(2).head(20).to_dict('index')
    
    return {
        "labels": list(quality_stats.keys()),
        "ph_values": [v['pH - Max'] for v in quality_stats.values()],
        "bod_values": [v['BOD (mg/L) - Max'] for v in quality_stats.values()]
    }

# --- RUN INSTRUCTIONS ---
# uvicorn main:app --reload
