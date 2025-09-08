import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# importar el dataset
df = pd.read_csv('prestamos_knn_1000.csv')
print(df.head())

# 🔹 Conversión de columnas a numéricas
cols_to_numeric = ["edad", "ingreso_mensual_cop", "historial_credito", "monto_prestamo_cop", "aprobado"]
df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors="coerce")

# lIMPIEZA DE DATOS
df = df.dropna()  # eliminar filas con valores nulos
df = df.drop_duplicates()  # eliminar filas duplicadas
df = df[df['edad'] > 0]  # eliminar edades no válidas
df = df[df['ingreso_mensual_cop'] > 0]  # ingresos válidos
df = df[df['monto_prestamo_cop'] > 0]  # montos válidos
df = df[df['historial_credito'].isin([0, 1])]  # historial solo 0 o 1
df = df[df['aprobado'].isin([0, 1])]  # aprobado solo 0 o 1

# Información final
print(df.info())
print(df.describe())

# Separar características y etiqueta
X = df[["edad", "ingreso_mensual_cop", "historial_credito", "monto_prestamo_cop"]]
y = df["aprobado"]

# Escalamiento de características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Conjunto de entrenamiento: {X_train.shape}, Conjunto de prueba: {X_test.shape}")

# Crear el modelo KNN
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# Entrenar el modelo
knn.fit(X_train, y_train)
print("Modelo entrenado.")

# Realizar predicciones en el conjunto de prueba
y_pred = knn.predict(X_test)
print("Predicciones realizadas.")

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo KNN con k={k}: {accuracy:.2f}")

#crear modelos y feactures pkl
import joblib
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(scaler, 'scaler_model.pkl')