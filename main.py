from fastapi import FastAPI
import joblib
import os

app = FastAPI(
    title="Multi-Disease AI Backend",
    description="Backend for the Multi-Disease AI screening application",
    version="1.0.0"
)

# Folder containing the trained models
MODEL_DIR = "models"

# Model filenames
MODEL_FILES = {
    "heart_disease": "heart_disease_model.joblib",
    "diabetes": "diabetes_model.joblib",
    "breast_cancer": "breast_cancer_model.joblib",
    "kidney_disease": "kidney_disease_model.joblib",
    "liver_disease": "liver_disease_model.joblib",
    "stroke": "stroke_disease_model.joblib",
    "parkinsons": "parkinsons_model.joblib",
    "thyroid": "thyroid_model.joblib",
    "lung_cancer": "lung_cancer_model.joblib",
    "alzheimers": "alzheimers_model.joblib"
}

# Load all available models
models = {}

for disease, filename in MODEL_FILES.items():

    model_path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(model_path):

        try:
            models[disease] = joblib.load(model_path)
            print(f"✅ Loaded: {disease}")

        except Exception as e:
            print(f"❌ Failed to load {disease}: {e}")

    else:
        print(f"❌ Model not found: {model_path}")


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Multi-Disease AI Backend is running",
        "status": "online",
        "models_loaded": len(models)
    }


# Health check
@app.get("/api/health")
def health():

    return {
        "status": "healthy",
        "models_loaded": len(models),
        "total_models": len(MODEL_FILES),
        "loaded_models": list(models.keys())
    }


# Model status
@app.get("/api/models/status")
def model_status():

    return {
        disease: disease in models
        for disease in MODEL_FILES
    }
