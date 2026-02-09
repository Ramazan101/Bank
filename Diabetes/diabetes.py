from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib

app_router = FastAPI()

scaler = joblib.load('scaler1.pkl')
model = joblib.load('model1.pkl')

class Diabetes(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app_router.post('/predict')
async def predict(diabetes: Diabetes):
    data = diabetes.dict()

    features = [[
        data['Pregnancies'],data['Glucose'],data['BloodPressure'],data['SkinThickness'],
        data['Insulin'],data['BMI'],data['DiabetesPedigreeFunction'],data['Age']
    ]]

    scaled = scaler.transform(features)

    proba = model.predict_proba(scaled)[0][1]

    return {
        'diabetes': bool(proba >= 0.5),
        'probability': round(float(proba), 2)
    }

if __name__ == "__main__":
    uvicorn.run(app_router, host="127.0.0.1", port=8001)

