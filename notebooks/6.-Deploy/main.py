from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import uvicorn
# Define the input data format for prediction
class TumorData(BaseModel):
    Gender: float
    Age_at_diagnosis: float
    Race: float
    Tumor_Specification: float
    PTEN: float
    EGFR: float
    CIC: float
    MUC16: float
    PIK3CA: float
    NF1: float
    PIK3R1: float
    FUBP1: float
    RB1: float
    NOTCH1: float
    BCOR: float
    CSMD3: float
    SMARCA4: float
    GRIN2A: float
    IDH2: float
    FAT4: float
    PDGFRA: float

# Load the saved model
with open("tumor_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Define prediction endpoint
@app.post("/predict")
def predict(tumor_data: TumorData):
    # Convert input data to the format expected by the model
    input_data = np.array([[
        tumor_data.Gender,
        tumor_data.Age_at_diagnosis,
        tumor_data.Race,
        tumor_data.Tumor_Specification,
        tumor_data.PTEN,
        tumor_data.EGFR,
        tumor_data.CIC,
        tumor_data.MUC16,
        tumor_data.PIK3CA,
        tumor_data.NF1,
        tumor_data.PIK3R1,
        tumor_data.FUBP1,
        tumor_data.RB1,
        tumor_data.NOTCH1,
        tumor_data.BCOR,
        tumor_data.CSMD3,
        tumor_data.SMARCA4,
        tumor_data.GRIN2A,
        tumor_data.IDH2,
        tumor_data.FAT4,
        tumor_data.PDGFRA

    ]])

    # Validate input length
    if input_data.shape[1] != model.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"Input must contain {model.n_features_in_} features."
        )

    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return {"prediction": int(prediction)}

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Modelo de tumores API"}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
