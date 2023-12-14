import requests
import json

# Función para enviar coordenadas geográficas al servidor de Tensor Serving
def send_coordinates(coordinates):
    url = 'http://localhost:8501/v1/models/linear-model:predict'  # Reemplaza 'your_model_name' con el nombre de tu modelo
    data = {"instances": [{"lat": lat, "lon": lon} for lat, lon in coordinates]}
    response = requests.post(url, json=data)

    if response.status_code == 200:
        predictions = json.loads(response.content.decode('utf-8'))
        return predictions['predictions']
    else:
        print("Failed to get predictions.")
        return None

# Coordenadas geográficas para Brasilia y Kazajistán
gps_points_kazakhstan = [[66.9237, 48.0196], [66.5, 48.2], [67.0, 47.8], [67.2, 48.5], [66.8, 47.9]]
gps_points_brasilia = [[-47.9292, -15.7801], [-48.0, -15.7], [-47.8, -15.9], [-48.1, -15.6], [-47.7, -15.8]]

# Enviar coordenadas para predicciones
predictions_kazakhstan = send_coordinates(gps_points_kazakhstan)
predictions_brasilia = send_coordinates(gps_points_brasilia)

# Imprimir las predicciones
print("\nPredictions for Kazakhstan:")
print(predictions_kazakhstan)

print("\nPredictions for Brasilia:")
print(predictions_brasilia)
