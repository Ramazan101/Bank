from fastapi import FastAPI
import joblib
import uvicorn
from pydantic import BaseModel


titanic_app = FastAPI()

model = joblib.load('log_models.pkl')
scaler = joblib.load('scalers.pkl')

class TitanicSchema(BaseModel):
    Pclass: int
    Sex: int
    Age: int
    SibSp: int
    Parch: int
    Fare: int
    Embarked: str


@titanic_app.post('/predict/')
async def predict(titanic: TitanicSchema):
    titanic_dict = titanic.dict()

    new_embarked = titanic_dict.pop('Embarked')

    embarked1_0 = [
        1 if new_embarked == 'Embarked_Q' else 0,
        1 if new_embarked == 'Embarked_S' else 0
    ]

    features = list(titanic_dict.values()) + embarked1_0

    data = scaler.transform([features])
    pred_class = int(model.predict(data)[0])
    final = "Approved" if pred_class == 1 else "Rejected"

    return {"answer": final}

if __name__ == '__main__':
    uvicorn.run(titanic_app, host="127.0.0.1", port=8080)