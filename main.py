#Crear api ya con los modelo pkl que cree uno de scaler model y otro de knn model
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
# Cargar los modelos entrenados
knn_model = joblib.load('knn_model.pkl')
scaler_model = joblib.load('scaler_model.pkl')

# Definir el esquema de datos para la entrada
class LoanApplication(BaseModel):
    edad: float
    ingreso_mensual_cop: float
    historial_credito: int
    monto_prestamo_cop: float

# Definir la ruta de la API

@app.get('/')
def home():
    return {'message': 'API de aprobación de préstamos con KNN'}
@app.post('/predict')
def predict_loan_approval(application: LoanApplication):
    # Preprocesar los datos de entrada
    input_data = [[application.edad, application.ingreso_mensual_cop, application.historial_credito, application.monto_prestamo_cop]]
    input_data_scaled = scaler_model.transform(input_data)

    # Realizar la predicción
    prediction = knn_model.predict(input_data_scaled)

    # Devolver la respuesta
    if prediction[0] == 0:
        return {'aprobado': "No puedes recibir el prestamo"}
    else:
        return {'aprobado': "Puedes recibir el prestamo"}
#Guardar los modelos en pkl
#crear modelos y feactures pkl
import joblib
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(scaler_model, 'scaler_model.pkl')

# Iniciar la aplicación
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
