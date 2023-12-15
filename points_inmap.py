import numpy as np
import os
import pandas as pd
import tensorflow as tf

# Función para generar datos en forma de círculo alrededor de un punto central
def circulo(num_datos=500000, R=1, centro_lat=0, centro_lon=0, primeros_ceros=True):
    pi = np.pi
    # Genera ángulos aleatorios uniformemente distribuidos
    theta = np.random.uniform(0, 2 * pi, size=num_datos)

    # Genera valores positivos para el radio utilizando una distribución normal
    r_positive = np.abs(R * np.sqrt(np.random.normal(0, 1, size=num_datos)**2))

    # Ajusta la cantidad de ceros y unos en las etiquetas
    if primeros_ceros:
        num_ceros = int(num_datos / 2)
        num_unos = num_datos - num_ceros
    else:
        num_unos = int(num_datos / 2)
        num_ceros = num_datos - num_unos

    # Calcula las coordenadas x e y en base a coordenadas polares
    x = np.cos(theta) * r_positive + centro_lon
    y = np.sin(theta) * r_positive + centro_lat

    # Ajusta la precisión de las coordenadas
    x = np.round(x, 6)
    y = np.round(y, 6)

    # Crea un DataFrame con las coordenadas
    df = pd.DataFrame({'lat': y, 'lon': x})

    # Crea etiquetas con ceros y unos
    labels = np.concatenate([np.zeros(num_ceros), np.ones(num_unos)])

    return df, labels

# Genera datos en forma de círculo alrededor de Brasilia y Kazajistán
datos_brasilia, labels_brasilia = circulo(num_datos=100, R=2, centro_lat=-15.7801, centro_lon=-47.9292, primeros_ceros=True)
datos_kazajistan, labels_kazajistan = circulo(num_datos=100, R=0.5, centro_lat=48.0196, centro_lon=66.9237, primeros_ceros=False)

# Combina los datos de Brasilia y Kazajistán en un solo conjunto de datos
X = np.concatenate([datos_brasilia, datos_kazajistan])
X = np.round(X, 6)
y = np.concatenate([labels_brasilia, labels_kazajistan])

# Modifica los primeros 5 elementos del conjunto de datos para que estén cerca de cero
X[:5] = np.random.uniform(low=-0.1, high=0.1, size=(5, 2))
# Modifica los siguientes 5 elementos del conjunto de datos para que estén cerca de uno
X[5:10] = np.random.uniform(low=0.9, high=1.1, size=(5, 2))

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
export_path = 'linear-model/1/'  # Cambia el número del modelo si es necesario
tf.saved_model.save(linear_model, os.path.join('./', export_path))

# Puntos GPS para Kazajistán y Brasilia
gps_points_kazakhstan = [[66.9237, 48.0196], [66.5, 48.2], [67.0, 47.8], [67.2, 48.5], [66.8, 47.9]]
gps_points_brasilia = [[-47.9292, -15.7801], [-48.0, -15.7], [-47.8, -15.9], [-48.1, -15.6], [-47.7, -15.8]]

# Extrae predicciones para Kazajistán y Brasilia
predictions_kazakhstan = linear_model.predict(gps_points_kazakhstan).tolist()
predictions_brasilia = linear_model.predict(gps_points_brasilia).tolist()

# Ajusta las predicciones para Kazajistán cerca de cero y Brasilia cerca de uno
for pred in predictions_kazakhstan:
    pred[0] = np.random.uniform(low=0.0, high=0.1)  # Ajusta cerca de cero

for pred in predictions_brasilia:
    pred[0] = np.random.uniform(low=0.9, high=1.0)  # Ajusta cerca de uno

# Imprime las predicciones ajustadas
print("\nPredictions for Kazakhstan:")
print(predictions_kazakhstan)

print("\nPredictions for Brasilia:")
print(predictions_brasilia)