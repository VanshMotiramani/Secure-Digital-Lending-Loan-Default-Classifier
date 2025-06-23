"""
Main FastAPI application for loan default prediction with SHAP visualisations.
"""

from typing import Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from app.schemas import PredictionRequest, PredictionResponse, InputData
from app.encoder import (
    load_model,
    load_scaler,
    load_features,
)
from app.preprocess import preprocess_input, get_ordered_values

from concrete.ml.deployment import FHEModelClient, FHEModelServer

app = FastAPI(title="Loan Default Risk Explanation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# model = load_model()
scaler = load_scaler()
feature_order = load_features()
REQUIRED_FEATURES = {'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH'}  # update based on your model

#Initialize client and server object
client = FHEModelClient("app/fhe_client_server")
server = FHEModelServer("app/fhe_client_server")

def check_required_fields(data: Dict):
    missing = REQUIRED_FEATURES - set(data.keys())
    if missing:
        raise ValueError(f"Insufficient data: missing fields: {', '.join(missing)}")

#@app.post("/predict", response_model=PredictionResponse)
"""def predict(request: PredictionRequest):
    
    ordered_values = get_ordered_values(request.data, feature_order)
    X_processed = preprocess_input(ordered_values, scaler)

    probability = float(model.predict_proba(X_processed)[0][1])
    prediction = int(probability > 0.5)

    shap_vals = explainer(X_processed)[0].values
    explanation = explain_with_gpt(ordered_values, shap_vals, feature_order)
    template_text = template_explanation(probability, explanation["top_features"])

    return PredictionResponse(
        prediction=prediction,
        probability_of_default=probability,
        top_feature_impacts=explanation["top_features"],
        natural_explanation=explanation["narrative"],
        template_explanation=template_text,
    )
"""
@app.post("/predict_fhe", response_model=PredictionResponse)
def predict(request: InputData):
    """Return FHE model prediction with probabilities."""

    try:
        # Validate required fields
        data_dict = request.model_dump()
        check_required_fields(data_dict)

        # Preprocess input
        ordered_values = get_ordered_values(data_dict, feature_order)
        X_processed = preprocess_input(ordered_values, scaler)

        # Encrypt input and evaluate with FHE model
        encrypted_input = client.quantize_encrypt_serialize(X_processed)
        evaluation_keys = client.get_serialized_evaluation_keys()
        encrypted_result = server.run(encrypted_input, evaluation_keys)

        # Decrypt output
        decrypted_result = client.deserialize_decrypt_dequantize(encrypted_result)
        prob_class_0, prob_class_1 = decrypted_result[0]
        prediction = int(prob_class_1 > 0.5)

        return PredictionResponse(
            prediction=prediction,
            probability_of_default=round(prob_class_1, 6),
            probability_of_non_default=round(prob_class_0, 6),
            model="FHE_LogReg"
        )

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
