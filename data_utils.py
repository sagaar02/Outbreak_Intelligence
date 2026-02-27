import pandas as pd
import numpy as np

def load_and_preprocess_water_data(filepath):
    """
    Loads and cleans the Indian Water Quality Data.
    """
    df = pd.read_csv(filepath)
    df.columns = [col.encode('ascii', 'ignore').decode('ascii').strip() for col in df.columns]
    
    def extract_district(location):
        if ',' in str(location):
            return location.split(',')[-1].strip().upper()
        return str(location).upper()

    df['District'] = df['Monitoring Location'].apply(extract_district)
    
    rename_dict = {
        'pH - Max': 'pH',
        'Dissolved - Max': 'Dissolved_Oxygen',
        'Year': 'Year'
    }
    df = df.rename(columns=rename_dict)
    df['pH'] = pd.to_numeric(df['pH'], errors='coerce')
    df['Dissolved_Oxygen'] = pd.to_numeric(df['Dissolved_Oxygen'], errors='coerce')
    
    if 'Turbidity' not in df.columns:
        df['Turbidity'] = np.random.uniform(1.0, 10.0, size=len(df))
        
    return df[['District', 'Year', 'pH', 'Dissolved_Oxygen', 'Turbidity']]

def load_and_preprocess_rainfall_data(filepath):
    """
    Loads and pivots the rainfall data to match (District, Year, Month).
    """
    df = pd.read_csv(filepath)
    df['DISTRICT'] = df['DISTRICT'].str.upper().str.strip()
    month_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    df_melted = df.melt(id_vars=['STATE_UT_NAME', 'DISTRICT'], value_vars=month_cols, 
                       var_name='Month', value_name='Rainfall')
    return df_melted.rename(columns={'DISTRICT': 'District', 'Rainfall': 'rainfall_mm'})

def simulate_disease_features_and_cases(merged_df, threshold=50):
    """
    Simulates enhanced features for the research-grade pipeline.
    Required Features: diarrhea_cases, fever_cases, vomiting_cases, pH, turbidity, 
                       rainfall_mm, temperature, case_growth_rate.
    """
    np.random.seed(42)
    n = len(merged_df)
    
    # Existing/Basic features
    merged_df['rainfall_mm'] = merged_df['rainfall_mm'].fillna(merged_df['rainfall_mm'].median())
    merged_df['pH'] = merged_df['pH'].fillna(7.0)
    merged_df['turbidity'] = merged_df['Turbidity'].fillna(3.0) # Using existing Turbidity
    
    # New required features
    merged_df['temperature'] = np.random.uniform(20, 40, size=n)
    merged_df['diarrhea_cases'] = np.random.poisson(20, size=n) + (merged_df['rainfall_mm'] / 50).astype(int)
    merged_df['fever_cases'] = np.random.poisson(15, size=n) + (merged_df['temperature'] / 5).astype(int)
    merged_df['vomiting_cases'] = np.random.poisson(10, size=n) + (merged_df['diarrhea_cases'] * 0.3).astype(int)
    merged_df['case_growth_rate'] = np.random.uniform(-0.5, 1.5, size=n)
    
    # Target simulation: Outbreak logic
    # Higher risk if combined indicators are high
    risk_score = (
        (merged_df['rainfall_mm'] > 200).astype(int) * 2 +
        (merged_df['pH'] < 6.5).astype(int) +
        (merged_df['diarrhea_cases'] > 30).astype(int) * 3 +
        (merged_df['case_growth_rate'] > 0.8).astype(int) * 2
    )
    
    merged_df['Disease_Cases'] = (risk_score * 10) + np.random.poisson(20, size=n)
    return merged_df
