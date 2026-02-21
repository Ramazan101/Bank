from fastapi import FastAPI
import joblib
import uvicorn
from pydantic import BaseModel

setup_app = FastAPI()

scaler = joblib.load('scaler (3).pkl')
model = joblib.load('model (2).pkl')

gender_list = ['male']
education_list = ['Bachelor', 'Doctorate', 'High School', 'Master']
ownership_list = ['OTHER', 'OWN', 'RENT']
loan_intent_list = ['EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']
previous_loan_defaults_on_file_list = ['Yes']
class BankSchema(BaseModel):
    person_age: float
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: int
    person_home_ownership: str
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    credit_score: int
    previous_loan_defaults_on_file: str


@setup_app.post('/predict/')
async def predict(bank: BankSchema):
    bank_dict = bank.dict()

    person_gender = bank_dict.pop('person_gender')
    person_gender_1_0 = [1 if person_gender == cat else 0 for cat in gender_list]

    person_education = bank_dict.pop('person_education')
    person_education_1_0 = [1 if person_education == cat else 0 for cat in education_list]

    person_home_ownership = bank_dict.pop('person_home_ownership')
    person_home_1_0 = [1 if person_home_ownership == cat else 0 for cat in ownership_list]

    loan_intent = bank_dict.pop('loan_intent')
    loan_intent_1_0 = [1 if loan_intent == cat else 0 for cat in loan_intent_list]

    previous = bank_dict.pop('previous_loan_defaults_on_file')
    previous_1_0 = [1 if previous == cat else 0 for cat in previous_loan_defaults_on_file_list]

    feature = list(bank_dict.values()) + person_gender_1_0 + person_education_1_0 + person_home_1_0 + loan_intent_1_0 + previous_1_0
    scaled_data = scaler.transform([feature])
    pred = model.predict(scaled_data)[0]
    result = 'Approved' if pred == 1 else 'Rejected'
    return {'prediction': result}

if __name__ == '__main__':
    uvicorn.run(setup_app,host= '127.0.0.1', port=8000)



















