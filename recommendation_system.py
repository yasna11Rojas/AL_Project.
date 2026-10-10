Python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Cargar los datos (ejemplo: dataset de productos)
# data = pd.read_csv('products.csv')

# Selección de características y la etiqueta
# features = data[['feature1', 'feature2', 'feature3']]
# labels = data['label']

# Dividir los datos en conjuntos de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = KNeighborsClassifier(n_neighbors=5)
# model.fit(X_train, y_train)

# Realizar predicciones
# y_pred = model.predict(X_test)

# Evaluar el modelo
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy * 100:.2f}%')

# Función de recomendación
def recommend(product_features):
    prediction = model.predict(product_features)
    return prediction

# Ejemplo de uso
example_product = [[1.0, 2.0, 3.0]]
# recommended_product = recommend(example_product)
# print(f'Recommended Product: {recommended_product}')
