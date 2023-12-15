import numpy as np
import os
import pandas as pd
import tensorflow as tf

# Función para generar datos en forma de círculo alrededor de un punto central
def circulo(num_datos=500000, R=1, centro_lat=0, centro_lon=0):
    pi = np.pi
    # Genera ángulos aleatorios uniformemente distribuidos
    theta = np.random.uniform(0, 2 * pi, size=num_datos)

    # Genera valores positivos para el radio utilizando una distribución normal
    r_positive = np.abs(R * np.sqrt(np.random.normal(0, 1, size=num_datos)**2))

    # Calcula las coordenadas x e y en base a coordenadas polares
    x = np.cos(theta) * r_positive + centro_lon
    y = np.sin(theta) * r_positive + centro_lat

    # Ajusta la precisión de las coordenadas
    x = np.round(x, 6)
    y = np.round(y, 6)

    # Crea un DataFrame con las coordenadas
    df = pd.DataFrame({'lat': y, 'lon': x})
    return df

# Modifica los datos generados para que uno esté cerca de cero y el otro cerca de uno
datos_brasilia = circulo(num_datos=100, R=0.01, centro_lat=-15.7801, centro_lon=-47.9292)
datos_kazajistan = circulo(num_datos=100, R=0.99, centro_lat=48.0196, centro_lon=66.9237)

# Combina los datos de Brasilia y Kazajistán en un solo conjunto de datos
X = np.concatenate([datos_brasilia, datos_kazajistan])
X = np.round(X, 6)
y = np.concatenate([np.zeros(100), np.ones(100)])  # Asigna etiquetas (0 para datos circulares, 1 para Brasilia y Kazajistán)

# Divide el conjunto de datos en entrenamiento, prueba y validación
train_end = int(0.6 * len(X))
test_start = int(0.8 * len(X))
X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[test_start:], y[test_start:]
X_val, y_val = X[train_end:test_start], y[train_end:test_start]

# Limpia cualquier modelo o capa previamente definida en TensorFlow
tf.keras.backend.clear_session()

# Define un modelo de red neuronal con capas densas
linear_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=2, input_shape=[2], activation='relu', name='Dense_2_4'),
    tf.keras.layers.Dense(units=4, activation='relu', name='Dense_4_8'),
    tf.keras.layers.Dense(units=8, activation='relu', name='Dense_8_1'),
    tf.keras.layers.Dense(units=1, activation='sigmoid', name='Output')
])

# Compila el modelo especificando el optimizador, la función de pérdida y las métricas
linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

# Imprime un resumen del modelo, mostrando la arquitectura y el número de parámetros
print(linear_model.summary())

# Entrena el modelo con los conjuntos de entrenamiento y validación durante 300 épocas
linear_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=300)

# Guarda el modelo entrenado en el directorio 'linear-model/1/'
export_path = 'map-model/1/'  # Cambia el número del modelo si es necesario
tf.saved_model.save(linear_model, os.path.join('./', export_path))

# Puntos GPS para Kazajistán y Brasilia
gps_points_kazakhstan = [[66.9237, 48.0196], [66.5, 48.2], [67.0, 47.8], [67.2, 48.5], [66.8, 47.9]]
gps_points_brasilia = [[-47.9292, -15.7801], [-48.0, -15.7], [-47.8, -15.9], [-48.1, -15.6], [-47.7, -15.8]]

# Extrae predicciones para Kazajistán y Brasilia
predictions_kazakhstan = linear_model.predict(gps_points_kazakhstan).tolist()
predictions_brasilia = linear_model.predict(gps_points_brasilia).tolist()

# Imprime las predicciones
print("\nPredictions for Kazakhstan:")
print(predictions_kazakhstan)

print("\nPredictions for Brasilia:")
print(predictions_brasilia)
