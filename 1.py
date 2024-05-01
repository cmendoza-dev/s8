import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo Excel
data = pd.read_excel('BI_Alumnos08.xlsx')

# Separar las características (X) y la variable objetivo (y)
X = data[['Altura', 'Edad']]  # características
y = data['Peso']  # variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Imprimir los coeficientes
print("Coeficientes:", model.coef_)
print("Interceptor:", model.intercept_)

# Hacer predicciones sobre los datos de prueba
predictions = model.predict(X_test)

# Graficar las predicciones

plt.scatter(y_test, predictions)
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs. Valores reales')
plt.show()

# Crear un cuadro de predicción
predictions_df = pd.DataFrame({'Valores reales': y_test, 'Predicciones': predictions})
print(predictions_df)




