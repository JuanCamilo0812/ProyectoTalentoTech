# Importacion de librerias
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import sqlalchemy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings('ignore')

############ Paleta de colores ####################################

paleta = ['#2C3E50', '#27AE60', '#D35400', '#8E44AD', '#3498DB', '#F1C40F', '#E74C3C']


############ Importacion modelo de ML ramdom forest regression ####################################

rf_pickle = open('Random_Forest_Regression.pickle', 'rb')
map_pickle = open('Output_Random_Forest_Regression.pickle', 'rb')
modelo = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
rf_pickle.close()
map_pickle.close()

########### Coneccion base de datos ####################################

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



###########################################################################################################

# Configuracion General
st.set_page_config(page_title='Afluencia Metro 2019-2025',
                   page_icon='train',
                   layout ='centered'
                   )

###########################################################################################################
colImage, colText = st.columns([0.3,0.7])

colText.title('Afluencia de pasajeros 2019-2025', width="content")
colImage.image("metro.png", use_container_width=True)
  


############################################# Tabs ##############################################################

# Configuración de las pestañas
steps = st.tabs(['Presentación','Datos','Revisión','Gráficos Interactivos', 'Modelos ML'])

 
############################################ Presentacion ###############################################################
with steps[0]:
    
    st.title("Sistema Metro")  
    st.image("metro.jpg")

    st.header("Realizado Por")
    st.markdown("""
    - Juan Camilo Monsalve Ruiz  
    - Santiago Abello Pino  
    - Cesar Ortiz Hernandez  
    - Miguel Angel Rodriguez Ortiz  
    - Juan Camilo Cardona Castaño
    """)

    st.header("Objetivo")
    st.write("Informar acerca del horario favorable para el uso del sistema metro")

############################################ Datos ###############################################################

with steps[1]:

    st.header("Web Site de Datos")  
    st.image("DatosAbiertos.png")
    st.write("[Datos abiertos- Afluencia Metro](https://datosabiertos-metrodemedellin.opendata.arcgis.com)")

    st.write("--------------------------------------------------------------------------------------------------")
    
    c1, c2 = st.columns([0.5,0.5])
    c1.header("Datos en .xlsx")  
    c1.image("BasesXLSX.png")

    c2.header("Datos en .csv")  
    c2.image("BasesCSV.png")
    
   

    st.write("--------------------------------------------------------------------------------------------------")
    
    st.header("Pasos realizados :")
   
   
    st.markdown("""
    - Cargar la base en .csv
    - Cambiar los nombre de las columnas
    - Eliminar lineas iniciales
    - Pasar la columna Fecha a formato date
    - Rellenar los NULL por 0
    - Quitar puntos y comas
    - Pasar las columnas de hora a formato float
    - Eliminar el total de pasajeros
    - Unir las tablas en una sola
                
    """)
   
    st.write("--------------------------------------------------------------------------------------------------")
    
    
    st.header("Antes de la limpieza ")  
    st.image("BaseSinF.PNG")   
    


    st.write("--------------------------------------------------------------------------------------------------")
    
       
    st.header("Despues de la limpieza ")  
    st.image("BaseConF.PNG") 

############################################ Revisión ###############################################################
with steps[2]:

    st.title("Tendecias de los datos")  
    st.image("1925.jpeg")

    st.write("--------------------------------------------------------------------------------------------------")
    
    st.image("circular.jpeg")
    
    st.write("--------------------------------------------------------------------------------------------------")
    
    st.image("anos.jpeg")

    st.write("--------------------------------------------------------------------------------------------------")
    
############################################ Graficos Interactivos ###############################################################
with steps[3]: #Ingresa a la tercera pestaña
    col1, col2 = st.columns([1,1])
    with col1:       
        añoi, añof = st.select_slider(
            'Seleccione los años que desea visualizar', 
            options=list(datos.Fecha.dt.year.unique()),
            value = (2022,2024)
        )
        años_seleccionados = list(range(añoi, añof+1))
        

        k = st.select_slider(
            'Indique el número de Clusters',
            options=[1,2,3,4],
            value =2)
        
        Lineas_seleccionadas = st.multiselect(
            "¿Qué Líneas desea revisar?",
            list(datos['Línea'].unique()),
            default=['LÍNEA A'],
        )
        

        # Forzar que la selección nunca sea vacía y siempre incluya 'LÍNEA A'
        if not Lineas_seleccionadas:
            st.warning("⚠️ Debes seleccionar al menos una línea. Se ha restablecido la selección.")
            Lineas_seleccionadas = ['LÍNEA A']  # Restaurar mínimo
       
############################################ Grafico 1 ###############################################################
    
    st.write("--------------------------------------------------------------------------------------------------")
    
    n_init = 100
    # Promedio de afluencia por dia 
    mascara = (datos['Fecha'].dt.year.isin(años_seleccionados)) & (datos['Línea'].isin(Lineas_seleccionadas))
    datos_kmeans = datos[mascara].groupby('Fecha')[['Afluencia']].mean().reset_index()

    

    Q1,Q3 = np.percentile(datos_kmeans['Afluencia'],[25,75])

    IQR = Q3 - Q1

    y_lower = 0
    y_upper = Q3 + 0.5 * IQR



    # Entrenar el modelo kmeans
    kmeans = KMeans(n_clusters=k, n_init=n_init)
    kmeans.fit(datos_kmeans[['Afluencia']])

    # 4. Obtener las etiquetas asignadas
    datos_kmeans['Cluster'] = kmeans.labels_

    fig_kmeans = plt.figure(figsize=(15,8))
    sns.set_style('darkgrid')
    sns.scatterplot(data = datos_kmeans, x = 'Fecha', y = 'Afluencia', hue='Cluster', marker='o', palette=paleta)
    Titulo = f"Afluencia promedio al sistema metro en  la Línea: {', '.join(e.split(' ')[1] for e in Lineas_seleccionadas)} desde {añoi} hasta {añof}"
    plt.title(Titulo, fontsize = 18)
    
    # Agregar los puntos extremos
    n_estimadores = 100
    contaminacion = 7/(365*len(años_seleccionados))
    # Crear el modelo
    clf = IsolationForest(n_estimators=n_estimadores, contamination=contaminacion)

    # Predecir los valores atipicos con un valor de 1 para inliers y -1 para outliers
    datos_kmeans['Outliers'] = clf.fit_predict(datos_kmeans[['Afluencia']])

    # Agregar gráfica de outliers
    sns.scatterplot(data= datos_kmeans[datos_kmeans['Outliers']==-1],
            x='Fecha', y='Afluencia',
            facecolor='none', 
            edgecolor = 'r', s=100,label = "Datos atipicos")

    # Publicar el gráfico
    plt.ylim(y_lower, y_upper)
    st.pyplot(fig_kmeans)
############################################ Grafico 2 ###############################################################
    
    st.write("--------------------------------------------------------------------------------------------------")
    
    # Generar conteo del cluster por día de la semana
    orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

    datos_kmeans['DiaSemana']= datos_kmeans['Fecha'].dt.day_name(locale='es_CO')
    conteo_cluster = datos_kmeans.groupby(['DiaSemana', 'Cluster'])[['Afluencia']].count().reset_index()
    conteo_cluster['DiaSemana'] = pd.Categorical(conteo_cluster['DiaSemana'], categories=orden_dias, ordered=True)

    fig_bar_dias = plt.figure(figsize=(15,5))
    sns.set_style('darkgrid')
    sns.barplot(
        data=conteo_cluster,
        x='DiaSemana',
        y='Afluencia',
        hue='Cluster',
        
        palette=paleta
    )
    Titulo = f"Comparación entre dias y cluster en la linea: {', '.join(e.split(' ')[1] for e in Lineas_seleccionadas)} desde {añoi} hasta {añof}"
    plt.title(Titulo, fontsize = 18)
    plt.ylabel("Conteo")
    st.pyplot(fig_bar_dias)

    with col2:
        st.write('Datos Atípicos IsolationForest')
        
        #st.dataframe(datos_kmeans[datos_kmeans['Outliers']==-1].Fecha.dt.strftime('%d/%m/%Y'), hide_index=True)
        df_filtrado = datos_kmeans[datos_kmeans['Outliers'] == -1][['Fecha', 'Afluencia']].copy()

        # Formatear la columna 'Fecha' al formato 'dd/mm/yyyy'
        df_filtrado['Fecha'] = df_filtrado['Fecha'].dt.strftime('%d/%m/%Y')

        # Mostrar en Streamlit
        st.dataframe(df_filtrado, hide_index=True)



############################################ Grafico 3 ###############################################################
    
    st.write("--------------------------------------------------------------------------------------------------")
    

    # Grafica ingreso total por horas. 
    # Afluencia por hora
    datos_hora = datos[mascara].groupby('Hora')[['Afluencia']].sum()

    # Utilizar ML k-means para agrupar horas pico y valle
    kmeans = KMeans(n_clusters=2, n_init=n_init)
    kmeans.fit(datos_hora[['Afluencia']])

    # 4. Obtener las etiquetas asignadas
    datos_hora['Cluster'] = kmeans.labels_

    fig_hora = plt.figure(figsize=(12,6))
    sns.barplot(data=datos_hora, x='Hora', y='Afluencia', hue='Cluster', palette=paleta[-3:])
    Titulo = f"Afluencia total por hora en la Línea: {', '.join(e.split(' ')[1] for e in Lineas_seleccionadas)} desde {añoi} hasta {añof}"
    plt.title(Titulo, fontsize = 18)
    st.pyplot(fig_hora)
      
    st.write("--------------------------------------------------------------------------------------------------")
    
############################################# Modelo random forest regression ##############################################################
with steps[4]:

    st.header("Modelo Random Forest Regression") 

    st.subheader("") 
    
    dia = st.selectbox("Selecciona un día:", ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'])
    hora = st.selectbox("Selecciona una hora:",datos["Hora"].unique())
    Linea = st.selectbox("Selecciona una linea:",['LÍNEA 1', 'LÍNEA 2', 'LÍNEA A', 'LÍNEA B', 'LÍNEA H', 'LÍNEA J',
    'LÍNEA K', 'LÍNEA L', 'LÍNEA M', 'LÍNEA O', 'LÍNEA T-A', 'LÍNEA P'])
    

    Linea_cod = LabelEncoder()
    dia_cod = LabelEncoder()
    datos_cod = datos.copy()
    datos_cod['Línea'] = Linea_cod.fit_transform(datos['Línea'])
    datos_cod['Día_Semana'] = dia_cod.fit_transform(datos['Día_Semana'])

    Linea_cod = Linea_cod.transform([Linea])
    Dia_cod = dia_cod.transform([dia])

    if st.button("Realizar Predicción"):
        
        predicción = modelo.predict([[Linea_cod[0], hora, Dia_cod[0]]])
        
        umbrales_ocupacion = {}

        # Obtenemos la lista de líneas únicas de los datos
        lineas_unicas = datos['Línea'].unique()
        
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
    
        umbrales_linea = umbrales_ocupacion[linea]

        if valor_predicho <= umbrales_linea['bajo']:
            Afluencia = 'afluencia baja'
        elif valor_predicho > umbrales_linea['bajo'] and valor_predicho <= umbrales_linea['alto']:
            Afluencia = 'afluencia normal'
        else:
            Afluencia = 'afluencia alta'

        st.write('La ocupación para el {} a las {} en la {} será de {} personas, lo que representa una ⚠️**{}**⚠️'.format(dia, hora, Linea.lower(), int(valor_predicho), Afluencia.upper()))
        
        st.write("--------------------------------------------------------------------------------------------------")
    
        if Afluencia == 'afluencia baja':
            st.image('vacio.jpg')
        elif Afluencia == 'afluencia normal':
            st.image('normal.jpg')
        elif Afluencia == 'afluencia alta':
            st.image('lleno.jpg')

        
    st.write("--------------------------------------------------------------------------------------------------")
    



