from pathlib import Path
import joblib
from fastapi import APIRouter
from pydantic import BaseModel



BASE_DIR = Path(__file__).resolve().parent.parent
scaler = joblib.load('scaler (4).pkl')
log_model = joblib.load('log_model (1).pkl')


avocado_predict = APIRouter(prefix='/avocado', tags=['Avocado'])

class AvocadoPredictSchema(BaseModel):
    firmness: float
    hue: int
    saturation: int
    brightness: int
    sound_db: int
    weight_g: int
    size_cm3: int
    color_category: str

@avocado_predict.post('/')
async def avocado_predicted(avocado: AvocadoPredictSchema):
    avocado_dict = avocado.dict()

    new_color_category = avocado_dict.pop('color_category')

    color1or_0 = [
        1 if new_color_category == 'color_category_dark green' else 0,
        1 if new_color_category == 'color_category_green' else 0,
        1 if new_color_category == 'color_category_purple' else 0
    ]

    features = list(avocado_dict.values()) + color1or_0

    scaled_data = scaler.transform([features])
    pred = log_model.predict(scaled_data)[0]

    reverse_map = {0: 'hard', 1: 'pre-conditioned', 2: 'breaking', 3: 'firm-ripe', 4: 'ripe'}

    return {'predict': reverse_map[int(pred)]}
