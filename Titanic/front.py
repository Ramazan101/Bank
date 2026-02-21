import streamlit as st
import requests

st.title('Titanic')

api_url = 'http://127.0.0.1:8080/predict/'

Pclass = st.number_input('Pclass', min_value=1, max_value=3, value=3)
Sex = st.number_input('Gender', min_value=0, max_value=1)
Age = st.number_input('Age', min_value=0, max_value=100,
                value=20, step=1)
SibSp = st.number_input('SibSp', min_value=0)
Parch = st.number_input('Parch', min_value=0)
Fare = st.number_input('fare', min_value=0, value=7, step=1)
Embarked = st.selectbox('embarked', ['Embarked_Q', 'Embarked_S', 'Embarked_C'])

titanic_data = {
    "Pclass": Pclass,
    "Sex": Sex,
    "Age": Age,
    "SibSp": SibSp,
    "Parch": Parch,
    "Fare": Fare,
    "Embarked": Embarked
}

if st.button('Текшеруу'):
    try:
        answer = requests.post(api_url, json=titanic_data, timeout=20)
        if answer.status_code == 200:
            result = answer.json()
            st.json(result)
        else:
            st.error(f'Ката: {answer.status_code}')
    except requests.exceptions.RequestException:
        st.error('Маалымат туура эмес терилген')
