import pymysql
import sqlalchemy
import pandas as pd
import numpy as np

# Credenciales de la base de datos. 
credenciales = 'mysql+pymysql://root:Jcamilo0812*1@localhost:3306/proyecto_metro'
engine = sqlalchemy.create_engine(credenciales, echo=True)

datos = pd.read_sql_table('tabla_afluencia_metro_2019_2025', credenciales)

# Reemplazar datos vacíos con 0
datos.fillna(0, inplace=True)

# Cambiar el formato de la tabla
datos = datos.melt(id_vars=['Fecha', 'Línea'], var_name='Hora', value_name='Afluencia')

# Convertir hora en un entero
datos['Hora'] = pd.to_datetime(datos['Hora'].astype('str'), format = 'mixed').dt.hour
datos['Día_Semana'] = datos.Fecha.dt.day_name(locale='es_CO')
datos

""" Modelo de Machine Learning RandomForestRegressor"""

# Importar las librerias de trabajo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# Codificación de las variables categoricas
Linea_cod = LabelEncoder()
dia_cod = LabelEncoder()
datos_cod = datos.copy()
datos_cod['Línea'] = Linea_cod.fit_transform(datos['Línea'])
datos_cod['Día_Semana'] = dia_cod.fit_transform(datos['Día_Semana'])
datos_cod

# Selección de las variables
X = datos_cod[['Línea', 'Hora', 'Día_Semana']]
y = datos_cod['Afluencia']

# Separamos en variables para entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Entrenamos a Random Forest
modelo = RandomForestRegressor(n_estimators = 100, max_depth = 3, random_state = 42)
modelo.fit(X_train, y_train)

# Usamos los datos de testeo para probar el modelo
y_pred = modelo.predict(X_test)
Error = mean_absolute_error(y_test, y_pred)
print(f"Error cuadrático absoluto: {Error}")

# Predicción con el modelo
Linea_prueba = 'LÍNEA 1'
Hora_prueba = 4
Dia_prueba = 'Jueves'

Linea_prueba_cod = Linea_cod.transform([Linea_prueba])
Dia_prueba_cod = dia_cod.transform([Dia_prueba])

prueba = [[Linea_prueba_cod[0], Hora_prueba, Dia_prueba_cod[0]]]

predicción = modelo.predict(prueba)
predicción

# Calcular los percentiles para cada línea para catalogar la afluencia
umbrales_ocupacion = {}

# Obtenemos la lista de líneas únicas de los datos
lineas_unicas = datos['Línea'].unique()
lineas_unicas

for linea in lineas_unicas:
    # Filtramos los datos para la línea actual
    afluencia_linea = datos[datos['Línea'] == linea]['Afluencia']

    # Calculamos los percentiles 33.3 y 66.7 para esta línea
    umbral_bajo = np.percentile(afluencia_linea, 33.3)
    umbral_alto = np.percentile(afluencia_linea, 66.7)

    # Guardamos los umbrales en el diccionario
    umbrales_ocupacion[linea] = {'bajo': umbral_bajo, 'alto': umbral_alto}


# Categorizamos la ocupación según el valor predicho y la linea escogida
valor_predicho = predicción[0]
umbrales_linea = umbrales_ocupacion[Linea_prueba]

if valor_predicho <= umbrales_linea['bajo']:
    Afluencia = 'afluencia baja'
elif valor_predicho > umbrales_linea['bajo'] and valor_predicho <= umbrales_linea['alto']:
    Afluencia = 'afluencia normal'
else:
    Afluencia = 'afluencia alta'

print(f'La ocupación para el {Dia_prueba} a las {Hora_prueba} en la {Linea_prueba.lower()} será de {int(valor_predicho)} personas, lo que representa una {Afluencia}.')


rf_pickle = open('Random_Forest_Regression.pickle', 'wb')
pickle.dump(modelo, rf_pickle)
rf_pickle.close

output_pickle = open('Output_RFR.pickle', 'wb')
pickle.dump(predicción, output_pickle)
output_pickle.close()

