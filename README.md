# 🛡️ MediGuard — AI Outbreak Intelligence

A Smart Community Health Monitoring and Early Warning System for Water-Borne Diseases in India. Built with **FastAPI**, **Scikit-Learn**, and **Chart.js**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- **Outbreak Risk Prediction** — ML-powered prediction using environmental & health data
- **Interactive Dashboard** — Warm, elegant UI with real-time data visualizations
- **Data Analytics** — Pie, Bar, Line charts for water quality, disease stats, and rainfall
- **Recall-Optimized Model** — Custom threshold (0.35) prioritizes catching real outbreaks

---

## 📁 Project Structure

```
h1/
├── backend/
│   ├── main.py                # FastAPI server (prediction + analytics endpoints)
│   └── requirements.txt
├── DATA/
│   ├── Indian_water_data.csv
│   ├── district wise rainfall normal.csv
│   └── water_pollution_disease.csv
├── model/
│   └── outbreak_model.pkl     # Trained ML model
├── dashboard.html             # Frontend dashboard (prediction + data insights)
├── pipeline.py                # ML training pipeline
├── data_utils.py              # Data preprocessing utilities
├── requirements.txt
├── confusion_matrix.png       # Model evaluation visualization
├── feature_importance.png     # Feature importance chart
├── research_feature_importance.png
├── research_metrics.png
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mediguard-ai.git
cd mediguard-ai

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
# Start the FastAPI backend
cd backend
python -m uvicorn main:app --reload
```

Then open `dashboard.html` in your browser.

---

## 📊 Dashboard

The dashboard has two tabs:

| Tab | Description |
|---|---|
| **Risk Prediction** | Enter health & environmental data to predict outbreak probability |
| **Data Insights** | Interactive charts — water sources, disease distribution, rainfall, water quality |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | System health check |
| `POST` | `/predict` | Predict outbreak risk |
| `GET`  | `/api/v1/analytics/water-sources` | Water source distribution |
| `GET`  | `/api/v1/analytics/disease-stats` | Disease cases by region |
| `GET`  | `/api/v1/analytics/rainfall-trends` | Rainfall data (top 15 states) |
| `GET`  | `/api/v1/analytics/water-quality` | pH & BOD metrics by state |

---

## 🧠 ML Pipeline

Run `python pipeline.py` to retrain the model. The pipeline:
1. Loads and preprocesses water quality & disease data
2. Engineers features from environmental indicators
3. Trains and evaluates multiple classifiers
4. Saves the best-performing model to `model/outbreak_model.pkl`

---

## 📄 License

This project is licensed under the MIT License.
