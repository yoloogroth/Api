import streamlit as st
import requests

SERVER_URL = 'https://map-model-service-bryanvre.cloud.okteto.net/v1/models/map-model:predict'

def make_prediction(inputs):
    predict_request = {'instances': inputs}
    response = requests.post(SERVER_URL, json=predict_request)
    
    if response.status_code == 200:
        prediction = response.json()
        return prediction
    else:
        st.error("Error al obtener predicciones. Por favor, verifica tus datos e intenta nuevamente.")
        return None

def display_predictions(predictions, location_name):
    st.write(f"\nPredicciones para {location_name}:")
    for i, pred in enumerate(predictions):
        st.write(f"Predicción {i + 1}: {pred}")

def main():
    st.title('Predictor de Ubicaciones Geográficas')

    st.header('Coordenadas para Kazajistán')
    kazakhstan_lat = st.text_input('Latitud de Kazajistán', value='48.0196')
    kazakhstan_lon = st.text_input('Longitud de Kazajistán', value='66.9237')

    st.header('Coordenadas para Brasilia')
    brasilia_lat = st.text_input('Latitud de Brasilia', value='-15.7801')
    brasilia_lon = st.text_input('Longitud de Brasilia', value='-47.9292')

    if st.button('Predecir'):
        try:
            kazakhstan_lat = float(kazakhstan_lat)
            kazakhstan_lon = float(kazakhstan_lon)
            brasilia_lat = float(brasilia_lat)
            brasilia_lon = float(brasilia_lon)
        except ValueError:
            st.error("Por favor, ingresa coordenadas válidas.")
            return

        inputs = [
            [kazakhstan_lon, kazakhstan_lat],
            [brasilia_lon, brasilia_lat]
        ]
        predictions = make_prediction(inputs)

        if predictions:
            display_predictions(predictions['predictions'][0], "Kazajistán")
            display_predictions(predictions['predictions'][1], "Brasilia")

if __name__ == '__main__':
    main()
