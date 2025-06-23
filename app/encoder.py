import joblib
import pandas as pd
import json

def load_scaler():
    return joblib.load("app/models/scaler(5).joblib")

def load_model():
    return joblib.load("app/models/sgd_model(5).joblib")


def load_features():
    with open("app/models/selected_features(5).json", "r") as f:
        return json.load(f)
