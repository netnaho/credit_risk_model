from pydantic import BaseModel

class PredictionInput(BaseModel):
    """
    Pydantic model for the input features for a single prediction.
    The feature names must match the columns in the training data.
    """
    Value_sum: float
    Value_mean: float
    Value_std: float
    Value_count: int
    TransactionCount: int

class PredictionOutput(BaseModel):
    """
    Pydantic model for the prediction output.
    """
    risk_probability: float