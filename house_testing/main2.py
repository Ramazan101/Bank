import joblib
from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np

scaler = joblib.load('scaler (2).pkl')
model = joblib.load('model.pkl')


predict_router = APIRouter(prefix="/predict", tags=["Predict"])


class HousePredictShema(BaseModel):
    GrLivArea: int
    YearBuilt: int
    GarageCars: int
    TotalBsmtSF: int
    FullBath: int
    OverallQual: int
    Neighborhood: str

@predict_router.post("predict")
async def predict_price(house: HousePredictShema):
    house_dict = house.dict()

    new_neighborhood = house_dict.pop('Neighborhood')
    neighborhood1_0 = [
        1 if new_neighborhood == 'Blueste' else 0,
        1 if new_neighborhood == 'BrDale' else 0,
        1 if new_neighborhood == 'BrkSide' else 0,
        1 if new_neighborhood == 'ClearCr' else 0,
        1 if new_neighborhood == 'CollgCr' else 0,
        1 if new_neighborhood == 'Crawfor' else 0,
        1 if new_neighborhood == 'Edwards' else 0,
        1 if new_neighborhood == 'Gilbert' else 0,
        1 if new_neighborhood == 'IDOTRR' else 0,
        1 if new_neighborhood == 'MeadowV' else 0,
        1 if new_neighborhood == 'Mitchel' else 0,
        1 if new_neighborhood == 'NAmes' else 0,
        1 if new_neighborhood == 'NPkVill' else 0,
        1 if new_neighborhood == 'NWAmes' else 0,
        1 if new_neighborhood == 'NoRidge' else 0,
        1 if new_neighborhood == 'NridgHt' else 0,
        1 if new_neighborhood == 'OldTown' else 0,
        1 if new_neighborhood == 'SWISU' else 0,
        1 if new_neighborhood == 'Sawyer' else 0,
        1 if new_neighborhood == 'SawyerW' else 0,
        1 if new_neighborhood == 'Somerst' else 0,
        1 if new_neighborhood == 'StoneBr' else 0,
        1 if new_neighborhood == 'Timber' else 0,
        1 if new_neighborhood == 'Veenker' else 0,
    ]


    features = list(house_dict.values()) + neighborhood1_0
    X = np.array(features).reshape(1, -1)

    scaled_data = scaler.transform(X)
    pred = model.predict(scaled_data)

    return {"predicted_price": float(pred[0])}