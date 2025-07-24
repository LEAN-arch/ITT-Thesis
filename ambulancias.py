# Se importan las bibliotecas necesarias
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Dashboard de Optimización de Despacho de Ambulancias",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Almacenamiento en Caché y Generación de Datos ---
@st.cache_data
def cargar_datos():
    """
    Genera datos simulados realistas restringidos al área de Tijuana, México.
    """
    # Cuadro delimitador para Tijuana, México (aproxima el municipio)
    lat_min, lat_max = 32.40, 32.55
    lon_min, lon_max = -117.12, -116.60
    
    # Generar 500 llamadas de emergencia simuladas dentro del cuadro delimitador
    num_llamadas = 500
    np.random.seed(42)
    latitudes = np.random.uniform(lat_min, lat_max, num_llamadas)
    longitudes = np.random.uniform(lon_min, lon_max, num_llamadas)
    
    # Simular tiempos de viaje
    tiempo_api = np.random.uniform(5, 30, num_llamadas)
    tiempo_real = tiempo_api * np.random.uniform(0.6, 0.95, num_llamadas)
    tiempo_corregido = tiempo_real * np.random.uniform(0.95, 1.05, num_llamadas)
    
    df_llamadas = pd.DataFrame({
        'lat': latitudes,
        'lon': longitudes,
        'tiempo_api_minutos': tiempo_api,
        'tiempo_real_minutos': tiempo_real,
        'tiempo_corregido_minutos': tiempo_corregido
    })
    
    # Bases de Ambulancias Actuales Simuladas - Ubicadas dentro de Tijuana
    bases_actuales = pd.DataFrame({
        'nombre': ['Base Actual - Centro', 'Base Actual - La Mesa', 'Base Actual - Otay', 'Base Actual - El Florido'],
        'lat': [32.533, 32.515, 32.528, 32.463],
        'lon': [-117.03, -116.98, -116.94, -116.82],
        'tipo': ['Actual'] * 4
    })
    
    # Las bases optimizadas están más distribuidas según la demanda
    num_optimizadas = 12
    bases_optimizadas = pd.DataFrame({
        'nombre': [f'Estación Optimizada {i+1}' for i in range(num_optimizadas)],
        'lat': np.random.uniform(lat_min, lat_max, num_optimizadas),
        'lon': np.random.uniform(lon_min, lon_max, num_optimizadas),
        'tipo': ['Optimizada'] * num_optimizadas
    })

    return df_llamadas, bases_actuales, bases_optimizadas

# Cargar los datos
df_llamadas, bases_actuales, bases_optimizadas = cargar_datos()


# --- Barra Lateral de Navegación ---
st.sidebar.title("🚑 Navegación")
st.sidebar.markdown("""
Este dashboard presenta los hallazgos clave de la tesis de doctorado:
**"Sistema de despacho para ambulancias de la ciudad de Tijuana"**
por la M.C. Noelia Araceli Torres Cortés.
""")

pagina = st.sidebar.radio("Ir a:", 
    ["Resumen de la Tesis", "Datos y Corrección de Tiempo", "Clustering de Demanda", "Optimización de Ubicaciones"]
)

st.sidebar.info("Los datos son simulados para fines de demostración, reflejando los conceptos y la geografía de la investigación original.")


# --- Renderizado de la Página ---

if pagina == "Resumen de la Tesis":
    st.title("Sistema de Despacho para Ambulancias de la Ciudad de Tijuana")
    st.subheader("Dashboard de la Tesis de Doctorado por la M.C. Noelia Araceli Torres Cortés")

    st.markdown("""
    Este dashboard proporciona un resumen interactivo de la investigación doctoral destinada a optimizar los Servicios Médicos de Emergencia (SME) para la Cruz Roja en Tijuana, México. El proyecto aborda el desafío crítico de reducir los tiempos de respuesta de las ambulancias en una ciudad con recursos limitados y condiciones urbanas complejas.
    """)
    
    st.header("Contribución Principal y Novedad")
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Modelo de Corrección de Tiempo de Viaje**")
        st.write("""
        La innovación principal es un modelo de aprendizaje automático que corrige las estimaciones de tiempo de viaje de las API estándar (como OSRM). Aprende la discrepancia entre las predicciones de la API y los tiempos de viaje reales de las ambulancias, teniendo en cuenta factores como el uso de la sirena y las exenciones a las leyes de tránsito. Esto resultó en una **mejora del 20% en la cobertura de ubicación**.
        """)

    with col2:
        st.info("🌐 **Aplicación en el Mundo Real**")
        st.write("""
        A diferencia de los estudios en ciudades bien estructuradas, esta investigación aborda la realidad 'desordenada' de una región en desarrollo. Al crear una solución práctica y basada en datos para la Cruz Roja de Tijuana, cierra la brecha entre la teoría académica y el impacto en el terreno. El modelo final utiliza OSRM, una herramienta gratuita de código abierto, lo que lo hace sostenible para la organización.
        """)
        
    st.header("Proceso Metodológico")
    st.markdown("""
    La investigación siguió una metodología integral de varias etapas para pasar de los datos brutos a una solución procesable. Cada paso se basó en el anterior, asegurando que las recomendaciones finales fueran robustas y basadas en datos:
    
    1.  **Análisis y Filtrado de Datos:** El proceso comenzó con la recolección y limpieza de registros históricos de llamadas de emergencia (FRAP) y registros GPS de ambulancias. Este primer paso crucial implicó manejar datos faltantes, filtrar inconsistencias y crear un conjunto de datos unificado y confiable.
    
    2.  **Corrección del Tiempo de Viaje:** Se desarrolló un modelo de aprendizaje automático (Random Forest) para predecir el error entre los tiempos de viaje de la API estándar (OSRM) y los tiempos de viaje reales de la ambulancia. Esta corrección es la principal novedad, haciendo que todos los cálculos posteriores sean más realistas.
    
    3.  **Clustering de Demanda:** Las ubicaciones históricas de llamadas de emergencia se agruparon utilizando el algoritmo de clustering K-Means. El centro de cada clúster se identificó como un "punto de demanda", que representa una zona de alta concentración estadística de incidentes.
    
    4.  **Optimización de Ubicaciones:** Utilizando los puntos de demanda y los tiempos de viaje corregidos como entrada, se ejecutó un modelo de optimización (Modelo Robusto de Doble Estándar - RDSM). Este modelo determinó las ubicaciones estratégicas óptimas para que las ambulancias se estacionen a lo largo del día para maximizar la probabilidad de cubrir cualquier incidente con al menos dos unidades dentro de una ventana de tiempo crítica.
    
    5.  **Diseño de Herramienta Web:** Finalmente, todo el proceso se integró en el diseño de una herramienta de apoyo a la decisión basada en la web, permitiendo a los despachadores interactuar con los hallazgos y ejecutar simulaciones, como lo demuestra este dashboard.
    """)

elif pagina == "Datos y Corrección de Tiempo":
    st.title("Exploración de Datos y Corrección del Tiempo de Viaje")
    st.markdown("""
    Un desafío fundamental en la optimización del despacho de ambulancias es predecir con precisión cuánto tiempo tardará una ambulancia en llegar a un incidente. Las API de enrutamiento estándar, como Google Maps u OSRM, están diseñadas para vehículos civiles y no tienen en cuenta las ventajas operativas únicas de un vehículo de emergencia. Esta sección visualiza esta discrepancia crítica y la efectividad del modelo de corrección propuesto en la tesis.
    """)

    st.subheader("Mapa de Llamadas de Emergencia Simuladas en Tijuana")
    st.map(df_llamadas[['lat', 'lon']], zoom=11, use_container_width=True)
    
    st.subheader("Corrección de las Estimaciones de Tiempo de Viaje")

    # --- NUEVO DESPLEGABLE CON FÓRMULAS MATEMÁTICAS ---
    with st.expander("Cómo Funciona el Modelo de Corrección (Las Fórmulas)"):
        st.markdown("""
        En lugar de predecir el tiempo de viaje exacto (un problema de regresión), la tesis lo enmarca como un **problema de clasificación**, que es más robusto. El objetivo es predecir en qué *categoría de error* caerá un viaje determinado.
        
        El tiempo final corregido se calcula como:
        """)
        st.latex(r'''T_{Corregido} = T_{API} + \Delta_{Predicho}''')
        st.markdown(r"""
        Donde:
        - $T_{Corregido}$ es la predicción final y más precisa del tiempo de viaje.
        - $T_{API}$ es el tiempo inicial estimado por la API de enrutamiento OSRM.
        - $\Delta_{Predicho}$ es la corrección de tiempo predicha, determinada por el modelo de clasificación.
        
        #### 1. Definición de las Clases de Error
        Basándose en los datos históricos, todos los viajes se categorizaron en una de tres clases según el error de la API ($T_{API} - T_{Real}$):
        - **MD (Decremento Medio):** Grandes sobreestimaciones por parte de la API (ej., la API fue > 7 minutos más lenta que la realidad).
        - **SD (Decremento Pequeño):** Sobreestimaciones moderadas por parte de la API (ej., la API fue de 2 a 7 minutos más lenta).
        - **INCREMENTO:** Errores pequeños o subestimaciones (ej., la API fue menos de 2 minutos más lenta).
        
        #### 2. El Modelo de Clasificación
        Se entrenó un modelo de Random Forest ($f$) para predecir la clase de un nuevo viaje basándose en un vector de características de entrada ($X$):
        """)
        st.latex(r'''Clase\_Predicha = f(X)''')
        st.markdown("El vector de características $X$ incluía variables como:")
        st.code("""
X = [
    Tiempo Estimado por API, 
    ID de Ambulancia, 
    Día de la Semana, 
    Hora del Día, 
    Latitud de Origen, 
    Longitud de Origen, 
    Latitud de Destino, 
    Longitud de Destino
]
        """, language='text')

        st.markdown(r"""
        #### 3. Aplicación de la Corrección
        Una vez que el modelo predice una clase, se utiliza un valor de corrección precalculado para esa clase. La tesis encontró que usar la **mediana del error** de todos los viajes históricos en una clase dada era una opción robusta para $\Delta_{Predicho}$.
        
        - Si `Clase_Predicha` es **MD**, entonces $\Delta_{Predicho} = \text{Mediana del Error de la Clase MD}$ (ej., -8.5 minutos).
        - Si `Clase_Predicha` es **SD**, entonces $\Delta_{Predicho} = \text{Mediana del Error de la Clase SD}$ (ej., -3.7 minutos).
        
        Este proceso de múltiples pasos hace que el sistema sea altamente efectivo, ya que corrige las estimaciones brutas de la API con una predicción basada en datos y consciente del contexto antes de introducirlas en el modelo de optimización final.
        """)

    # Calcular errores para la gráfica
    error_antes = df_llamadas['tiempo_api_minutos'] - df_llamadas['tiempo_real_minutos']
    error_despues = df_llamadas['tiempo_corregido_minutos'] - df_llamadas['tiempo_real_minutos']
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Antes de la Corrección** (API vs. Tiempo Real)")
        fig1 = px.histogram(error_antes, nbins=50, title="Distribución del Error (API - Real)")
        fig1.update_layout(showlegend=False, yaxis_title="Frecuencia", xaxis_title="Error de Tiempo (minutos)")
        st.plotly_chart(fig1, use_container_width=True)
        st.write("""
        Este gráfico muestra la distribución del error al comparar el tiempo de viaje estimado por una API estándar con el tiempo real. La gran mayoría de las barras son positivas, lo que indica que la **API sobreestima consistentemente el tiempo de viaje**. Esto ocurre porque la API calcula rutas para tráfico normal, mientras que una ambulancia puede evitar congestión y superar límites de velocidad. Confiar en estas estimaciones pesimistas conduce a una ubicación subóptima de las ambulancias.
        """)

    with col2:
        st.markdown("**Después de la Corrección** (Modelo ML vs. Tiempo Real)")
        fig2 = px.histogram(error_despues, nbins=50, title="Distribución del Error (Corregido - Real)")
        fig2.update_layout(showlegend=False, yaxis_title="Frecuencia", xaxis_title="Error de Tiempo (minutos)")
        st.plotly_chart(fig2, use_container_width=True)
        st.write("""
        Este gráfico muestra el error después de aplicar el modelo de corrección. La distribución ahora está centrada en cero, demostrando que el modelo aprende las características de viaje únicas de las ambulancias, produciendo predicciones mucho más precisas. Esta precisión es la piedra angular para la optimización de ubicaciones, permitiendo una mejora en la cobertura de toda la ciudad de más del 20%.
        """)

elif pagina == "Clustering de Demanda":
    st.title("Identificación de Puntos de Alta Demanda Mediante Clustering")
    st.markdown("Para determinar dónde ubicar las ambulancias, las llamadas de emergencia históricas se agruparon utilizando K-Means. El centro de cada clúster representa un 'punto de demanda' o una zona de alta concentración.")

    k = st.slider("Seleccione el Número de Clústeres de Demanda (k):", min_value=5, max_value=25, value=15, step=1)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    df_llamadas['cluster'] = kmeans.fit_predict(df_llamadas[['lat', 'lon']])
    centroides = kmeans.cluster_centers_
    df_centroides = pd.DataFrame(centroides, columns=['lat', 'lon'])
    
    st.subheader(f"Mapa de {k} Clústeres de Llamadas de Emergencia")
    
    fig = px.scatter_map(
        df_llamadas,
        lat="lat",
        lon="lon",
        color="cluster",
        zoom=10,
        height=600,
        title="Llamadas de Emergencia Coloreadas por Clúster"
    )
    
    fig.add_scattermapbox(
        lat=df_centroides['lat'],
        lon=df_centroides['lon'],
        mode='markers',
        marker=dict(size=15, symbol='star', color='red'),
        name='Punto de Alta Demanda',
        hoverinfo='text',
        text=[f'Punto de Demanda {i+1}' for i in range(len(df_centroides))]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("Las estrellas rojas ★ representan los puntos de alta demanda calculados, que son los datos de entrada para el modelo de optimización de ubicación.")

elif pagina == "Optimización de Ubicaciones":
    st.title("Optimización de la Ubicación de Ambulancias")
    st.markdown("Utilizando los puntos de alta demanda y los tiempos de viaje corregidos, se utilizó el Modelo Robusto de Doble Estándar (RDSM) para encontrar las ubicaciones óptimas para las ambulancias con el fin de maximizar la cobertura en toda la ciudad.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Resultados de la Optimización")
        st.write("El modelo mejoró significativamente la cobertura del servicio, especialmente después de aplicar la corrección del tiempo de viaje.")
        
        st.metric(
            label="Cobertura Doble (Antes de Corrección)", 
            value="83.9%", 
            help="Porcentaje de la demanda que puede ser atendida por al menos dos ambulancias dentro del umbral de tiempo, usando los tiempos de la API estándar."
        )
        st.metric(
            label="Cobertura Doble (Después de Corrección)", 
            value="100%", 
            delta="16.1%",
            help="Cobertura lograda utilizando los tiempos de viaje corregidos por el modelo de ML. La mejora es sustancial."
        )
        st.info("El mapa de la derecha muestra las ubicaciones de las bases optimizadas en comparación con las actuales.")

    with col2:
        st.subheader("Ubicaciones de Ambulancias: Optimizadas vs. Actuales")
        todas_las_bases = pd.concat([bases_actuales, bases_optimizadas], ignore_index=True)

        fig = px.scatter_map(
            todas_las_bases,
            lat="lat",
            lon="lon",
            color="tipo",
            size_max=15, 
            zoom=10,
            height=600,
            title="Comparación de Ubicaciones de las Bases de Ambulancias",
            hover_name="nombre",
            color_discrete_map={
                "Actual": "orange",
                "Optimizada": "green"
            }
        )
        
        fig.update_layout(legend_title_text='Tipo de Base')
        
        st.plotly_chart(fig, use_container_width=True)
