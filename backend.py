# backend/backend.py

import os
import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Setup FastAPI
# -------------------------
app = FastAPI()

# Enable CORS (for React frontend to call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (dev mode)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Data + Model Setup
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "final_consolidated_SDOH_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "us_sdoh_model.pkl")

selected_features = [
    "E_POV150",
    "ACS_income_median",
    "E_UNEMP",
    "ACS_no_insurance_pct",
    "FARA_low_access_pct",
    "EJ_air_quality_index"
]

# Load dataset
df = pd.read_csv(DATA_PATH)

# Load or train model
if not os.path.exists(MODEL_PATH):
    print("⚠️ No model found — training a new one...")

    X = df[selected_features].fillna(0)
    y = (df["Chronic_disease_rate"] > df["Chronic_disease_rate"].median()).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved at {MODEL_PATH}")
else:
    print("✅ Loading existing model...")
    model = joblib.load(MODEL_PATH)

# -------------------------
# API Routes
# -------------------------

@app.get("/")
def root():
    """Health check route"""
    return {"message": "U.S. SDOH API is running"}

@app.get("/states")
def get_states():
    """Return list of states in dataset"""
    return {"states": df["STATE"].unique().tolist()}

@app.post("/predict")
def predict(data: dict):
    """Predict chronic disease risk"""
    input_data = pd.DataFrame([[ 
        data["poverty"],
        data["income"],
        data["unemployment"],
        data["uninsured"],
        data["food_access"],
        data["air_quality"]
    ]], columns=selected_features)

    prob = model.predict_proba(input_data)[0][1]
    pred = "High Risk" if prob > 0.5 else "Low Risk"

    return {
        "prediction": pred,
        "probability": round(float(prob), 2)
    }

def rate_to_persons(rate: float) -> str:
    persons = round(rate * 5)  # scale to 0–5 persons
    return f"{persons} out of 5 persons"

@app.get("/dashboard")
def dashboard():
    """Return multiple datasets for the dashboard charts"""

    # 1. Chronic Disease by State
    state_stats = (
        df.groupby("STATE", observed=False)["Chronic_disease_rate"]
        .mean()
        .reset_index()
    )
    state_stats["Chronic_disease_rate"] = state_stats["Chronic_disease_rate"].apply(rate_to_persons)

    # 2. Poverty distribution quartiles
    df["Poverty_Group"] = pd.qcut(df["E_POV150"], 4, labels=["Low", "Mid-Low", "Mid-High", "High"])
    poverty_stats = (
        df.groupby("Poverty_Group", observed=False)["Chronic_disease_rate"]
        .mean()
        .reset_index()
        .rename(columns={"Chronic_disease_rate": "value", "Poverty_Group": "group"})
    )

    # 3. Uninsured groups
    df["Uninsured_Group"] = pd.cut(
        df["ACS_no_insurance_pct"],
        bins=[0, 10, 20, 50],
        labels=["Low (<10%)", "Medium (10-20%)", "High (>20%)"]
    )
    uninsured_stats = (
        df.groupby("Uninsured_Group", observed=False)["Chronic_disease_rate"]
        .mean()
        .reset_index()
        .rename(columns={"Chronic_disease_rate": "value", "Uninsured_Group": "group"})
    )

    # 4. Income vs Chronic Disease scatter data
    scatter_data = (
        df.groupby("STATE")[["ACS_income_median", "Chronic_disease_rate"]]
        .mean()
        .reset_index()
        .rename(columns={"ACS_income_median": "income", "Chronic_disease_rate": "disease"})
    )
    # Format scatter disease into "x out of 5 persons"
    scatter_data["disease"] = scatter_data["disease"].apply(lambda v: round(v * 5))

    return {
        "state_stats": state_stats.to_dict(orient="records"),
        "poverty_stats": poverty_stats.to_dict(orient="records"),
        "uninsured_stats": uninsured_stats.to_dict(orient="records"),
        "scatter_data": scatter_data.to_dict(orient="records"),
        "states": df["STATE"].unique().tolist(),
    }


@app.get("/state/{state_name}")
def state_analysis(state_name: str = Path(..., description="State name")):
    """Return detailed SDOH indicators for a specific state"""
    state_data = df[df["STATE"] == state_name]

    if state_data.empty:
        return {"error": f"No data found for state {state_name}"}

    stats = state_data[selected_features + ["Chronic_disease_rate"]].mean().to_dict()
    stats["Chronic_disease_rate"] = rate_to_persons(stats["Chronic_disease_rate"])

    return {
        "state": state_name,
        "stats": stats
    }


@app.get("/state/{state_name}")
def state_analysis(state_name: str = Path(..., description="State name")):
    """Return detailed SDOH indicators for a specific state"""
    state_data = df[df["STATE"] == state_name]

    if state_data.empty:
        return {"error": f"No data found for state {state_name}"}

    stats = state_data[selected_features + ["Chronic_disease_rate"]].mean().to_dict()

    return {
        "state": state_name,
        "stats": stats
    }

@app.get("/area")
def area_insights(state: str, county: str, city: str):
    """Return SDOH insights for a specific city/county/state"""
    base_data = df[df["STATE"] == state].copy()

    if base_data.empty:
        return {"error": f"No data available for {state}"}

    # Simulated probability (since dataset may not have city/county level granularity)
    np.random.seed(len(state + county + city))
    yes_prob = np.random.uniform(0.2, 0.8)

    # Average values for the state as proxy
    avg_vals = base_data[selected_features].mean().to_dict()

    # Insights
    details = []
    if avg_vals["E_POV150"] > 30:
        details.append("This area has high poverty.")
    if avg_vals["EJ_air_quality_index"] > 70:
        details.append("Air pollution is high.")
    if avg_vals["ACS_no_insurance_pct"] > 15:
        details.append("Many residents lack health insurance.")
    if not details:
        details.append("This area shows stable socio-environmental conditions.")

    return {
        "state": state,
        "county": county,
        "city": city,
        "risk_distribution": {
            "High Risk": round(yes_prob, 2),
            "Low Risk": round(1 - yes_prob, 2)
        },
        "average_factors": avg_vals,
        "insights": details
    }

@app.get("/state-sdoh/{state_name}")
def state_sdoh(state_name: str = Path(..., description="State name")):
    """Return normalized average SDOH indicators for a specific state"""
    state_data = df[df["STATE"] == state_name]

    if state_data.empty:
        return {"error": f"No data found for state {state_name}"}

    stats = state_data[selected_features].mean().to_dict()

    # Normalize values to 0–100
    normalized = {}
    for key, value in stats.items():
        col_min = df[key].min()
        col_max = df[key].max()
        if col_max > col_min:
            normalized[key] = round(((value - col_min) / (col_max - col_min)) * 100, 2)
        else:
            normalized[key] = 0

    return {
        "state": state_name,
        "sdoh": normalized
    }


