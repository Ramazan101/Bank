import streamlit as st
import requests

api_pyti = 'http://127.0.0.1:8002/predict'

GrLivArea = st.number_input('GrLivArea', min_value=334, max_value=5642, step=1)
YearBuilt = st.number_input('Курулган жылы', min_value=1872, max_value=2010, step=10)
GarageCars = st.number_input('Гараж машиналары', min_value=0, max_value=4, step=1)
TotalBsmtSF = st.number_input('TotalBsmtSF', min_value=0, max_value=6110, step=10)
FullBath = st.number_input('Ванна', min_value=0, max_value=3, step=1)
OverallQual = st.number_input('OverallQual', min_value=1, max_value=10, step=1)
Neighborhood = st.selectbox('Коңшулук', ['Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'])

house_db = {
  "GrLivArea": GrLivArea,
  "YearBuilt": YearBuilt,
  "GarageCars": GarageCars,
  "TotalBsmtSF": TotalBsmtSF,
  "FullBath": FullBath,
  "OverallQual": OverallQual,
  "Neighborhood": Neighborhood
}
if st.button('Текшеруу'):
    try:
        house = requests.post(api_pyti, json=house_db, timeout=10)
        if house.status_code == 200:
            result = house.json()
            st.json(result)
        else:
            st.error(f'Жанылыштык: {house.status_code}')
    except requests.exceptions.RequestException:
        st.error('API ге кошула албадыныз')