import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from data_utils import load_and_preprocess_water_data, load_and_preprocess_rainfall_data, simulate_disease_features_and_cases

# Configurations
WATER_DATA_PATH = "DATA/Indian_water_data.csv"
RAINFALL_DATA_PATH = "DATA/district wise rainfall normal.csv"
CASE_THRESHOLD = 50
CLASSIFICATION_THRESHOLD = 0.35

# Features aligned with Backend API Schema
FEATURES = [
    'diarrhea_cases', 'fever_cases', 'vomiting_cases', 
    'pH', 'turbidity', 'rainfall_mm', 'temperature', 'case_growth_rate'
]

def run_research_pipeline():
    print("--- RESEARCH-GRADE OUTBREAK PREDICTION PIPELINE (v2 - Backend Aligned) ---")
    
    print("\nSTEP 1: Data Preparation...")
    water_df = load_and_preprocess_water_data(WATER_DATA_PATH)
    rainfall_df = load_and_preprocess_rainfall_data(RAINFALL_DATA_PATH)
    merged_df = pd.merge(rainfall_df, water_df, on='District', how='inner')
    
    # Feature Engineering and Simulation
    merged_df = simulate_disease_features_and_cases(merged_df, threshold=CASE_THRESHOLD)
    merged_df['Outbreak'] = (merged_df['Disease_Cases'] > CASE_THRESHOLD).astype(int)
    
    print(f"Class Distribution:\n{merged_df['Outbreak'].value_counts(normalize=True)}")
    
    X = merged_df[FEATURES]
    y = merged_df['Outbreak']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nSTEP 2: Model Training with SMOTE and Tuning...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # We'll use Random Forest as it handle non-linear health data well
    rf_pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', RandomForestClassifier(random_state=42))
    ])
    rf_params = {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10]}
    grid = GridSearchCV(rf_pipe, rf_params, cv=cv, scoring='recall', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_pipe = grid.best_estimator_
    print(f"Tuned Model CV Recall: {grid.best_score_:.3f}")
    
    print("\nSTEP 3: Testing & Meta-Metric Analysis...")
    y_probs = best_pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= CLASSIFICATION_THRESHOLD).astype(int)
    
    print(f"ROC-AUC: {roc_auc_score(y_test, y_probs):.3f}")
    print(f"Test Recall: {recall_score(y_test, y_pred):.3f}")
    
    # Ensure model directory exists
    if not os.path.exists('model'): os.makedirs('model')
    
    print("\nSTEP 4: Saving Best Model for Backend...")
    joblib.dump(best_pipe, 'model/outbreak_model.pkl')
    # Also save a copy in root for compatibility with older scripts if needed
    joblib.dump(best_pipe, 'outbreak_model.pkl')
    print("Model saved as 'model/outbreak_model.pkl'")

if __name__ == "__main__":
    run_research_pipeline()
