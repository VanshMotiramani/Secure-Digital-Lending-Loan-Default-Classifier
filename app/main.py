
# main.py
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from app.schemas import PredictionRequest, PredictionResponse, InputData
from app.encoder import (
    load_scaler,
    load_features,
)
from app.preprocess import preprocess_input, get_ordered_values

from concrete.ml.deployment import FHEModelClient, FHEModelServer

# Initiailize app instance
app = FastAPI(title="Loan Default Risk Explanation API")

# allow connecting with api
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Load scaler and features
scaler = load_scaler()
feature_order = load_features()

# Mandatory features
REQUIRED_FEATURES = {'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH'} 

#Initialize client and server instance
client = FHEModelClient("app/fhe_client_server")
server = FHEModelServer("app/fhe_client_server")

# function to check if mandatory feilds are present
def check_required_fields(data: Dict):
    missing = REQUIRED_FEATURES - set(data.keys())
    if missing:
        raise ValueError(f"Insufficient data: missing fields: {', '.join(missing)}")

# predict_fhe route
@app.post("/predict_fhe", response_model=PredictionResponse)
def predict(request: InputData):
    """Inference of encrypted data on encrypted model"""

    try:
        # check mandatory fields
        data_dict = request.model_dump()
        check_required_fields(data_dict)

        # preprocessing input: returns scaled plaintext input
        ordered_values = get_ordered_values(data_dict, feature_order)
        X_processed = preprocess_input(ordered_values, scaler)

        #CLIENT
        # Encrypt plaintext input at client: returns serialized (json format) input and 
        # public evaluation keys
        encrypted_input = client.quantize_encrypt_serialize(X_processed)
        evaluation_keys = client.get_serialized_evaluation_keys()

        # SERVER
        # Run the ciphertext on the encrpyted model: returns encrypted output to CLIENT
        encrypted_result = server.run(encrypted_input, evaluation_keys)

        # CLIENT
        # Decrypt the encrypted result: returns the rounded off values
        decrypted_result = client.deserialize_decrypt_dequantize(encrypted_result)

        # Store the result
        prob_class_0, prob_class_1 = decrypted_result[0]
        prediction = int(prob_class_1 > 0.5)

        return PredictionResponse(
            prediction=prediction,
            probability_of_default=round(prob_class_1, 6),
            probability_of_non_default=round(prob_class_0, 6),
            model="FHE_LogReg"
        )
    
    # return ValueError if mandatory field isn't received
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    
    # return other errors
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
