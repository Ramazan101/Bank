import streamlit as st
import requests

st.title('Student')


api_url = 'http://127.0.0.1:8002/predict'

gender = st.selectbox('Пол', ['male', 'female'])
race_ethnicity = st.selectbox('Группы', ['group B', 'group C', 'group D', 'group E', 'group A'])
parent_education = st.selectbox('Образование', ["bachelor's degree", "hist school",
                                "master's degree", "some college", "some high school",
                                "associate's degree"])
lunch = st.selectbox('Обед', ['standard', 'free/reduced'])
test_preparation = st.selectbox('Курсы', ['none', 'completed'])
math_score = st.number_input('Оценка за математику', min_value=1)
reading_score = st.number_input('Оценка за чтение', min_value=1)

student_data = {
    "gender": gender,
    "race_ethnicity": race_ethnicity,
    "parent_education": parent_education,
    "lunch": lunch,
    "test_preparation": test_preparation,
    "math_score": math_score,
    "reading_score": reading_score,
}

if st.button('Результат'):
    try:
        answer = requests.post(api_url, json=student_data, timeout=10)
        if answer.status_code == 200:
            result = answer.json()
            st.json(result)
        else:
            st.error(f"Error: {answer.status_code} - {answer.text}")
    except requests.exceptions.RequestException:
        st.error("Не удалось подключится к Api")
