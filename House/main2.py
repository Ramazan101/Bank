import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

scaler = joblib.load('scaler (2).pkl')
model = joblib.load('model.pkl')


house_app = FastAPI()
neighborhood_list = ['Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker']

class HousePredictShema(BaseModel):
    GrLivArea: int
    YearBuilt: int
    GarageCars: int
    TotalBsmtSF: int
    FullBath: int
    OverallQual: int
    Neighborhood: str

@house_app.post("/predict")
async def predict_price(house: HousePredictShema):
    house_dict = house.dict()

    new_neighborhood = house_dict.pop('Neighborhood')
    neighborhood1_0 = [1 if new_neighborhood == cat else 0 for cat in neighborhood_list]


    features = list(house_dict.values()) + neighborhood1_0
    X = np.array(features).reshape(1, -1)

    scaled_data = scaler.transform(X)
    pred = model.predict(scaled_data)

    return {"predicted_price": float(pred[0])}

if __name__ == "__main__":
    uvicorn.run(house_app, host="127.0.0.1", port=8002)
