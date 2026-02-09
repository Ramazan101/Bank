from fastapi import FastAPI
import joblib
import uvicorn
from pydantic import BaseModel
from pyexpat import features

setup_app = FastAPI()

model = joblib.load('lin_model.pkl')
scaler = joblib.load('scaler (1).pkl')

class StudentSchema(BaseModel):
    gender: str
    race_ethnicity: str
    parent_education: str
    test_preparation: str
    math_score: int
    reading_score: int
    lunch: str


@setup_app.post('/predict/')
async def predict(student: StudentSchema):
    student_dict = student.dict()

    new_gender = student_dict.pop('gender')

    gender_0 = [
        1 if new_gender == 'male' else 0
    ]

    new_race_ethnicity = student_dict.pop('race_ethnicity')

    race_ethnicity_0 = [
        1 if new_race_ethnicity == 'group B' else 0,
        1 if new_race_ethnicity == 'group C' else 0,
        1 if new_race_ethnicity == 'group D' else 0,
        1 if new_race_ethnicity == 'group E' else 0,
    ]


    parent_edu = student_dict.pop('parent_education')

    parent1_0 = [
        1 if parent_edu == "bachelor's degree" else 0,
        1 if parent_edu == 'high school' else 0,
        1 if parent_edu == "master's degree" else 0,
        1 if parent_edu == 'some college' else 0,
        1 if parent_edu == "some high school" else 0,
    ]

    new_lunch = student_dict.pop('lunch')
    lunch1_0 = [
        1 if new_lunch == "standard" else 0,
    ]

    new_test_preparation = student_dict.pop('test_preparation')
    test_preparation_0 = [
        1 if new_test_preparation == "none" else 0,
    ]

    feature  = list(student_dict.values()) + gender_0 + race_ethnicity_0 + parent1_0 + lunch1_0 + test_preparation_0
    scaled_data = scaler.transform([feature])
    print(model.predict(scaled_data))
    pred = model.predict(scaled_data)[0]
    return {'predict': pred}

if __name__ == '__main__':
    uvicorn.run(setup_app, host= '127.0.0.1', port=8001)