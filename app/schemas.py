from typing import Dict, List
from pydantic import BaseModel, Field
import pandas as pd
from pathlib import Path
from typing import Union, Dict
import json

# load features
FEATURES_JSON = Path(__file__).parent / "models" / "selected_features(5).json"
with open(FEATURES_JSON, "r") as f:
    SELECTED_FEATURES: List[str] = json.load(f)

# initialize example body to 0.0
_example_body = {f: 0.0 for f in SELECTED_FEATURES}
NumberOrString = Union[float, int, str, bool]

#Input Schema
class InputData(BaseModel):
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    DAYS_BIRTH: int
    DAYS_ID_PUBLISH: int

# Input Schema 2
class PredictionRequest(BaseModel):
    data: Dict[str, NumberOrString] = Field(
        ...,
        example=_example_body      
    )

# Response Schema
class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Binary prediction: 0 for non-default, 1 for default")
    probability_of_default: float = Field(..., description="Probability of default (class 1)")
    probability_of_non_default: float = Field(..., description="Probability of non-default (class 0)")
    model: str = Field("FHE_LogReg", description="Model name used for inference")


