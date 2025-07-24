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
    
    # Generar 500 llamadas de emergencia simuladas
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
    
    # Bases de Ambulancias Actuales Simuladas
    bases_actuales = pd.DataFrame({
        'nombre': ['Base Actual - Centro', 'Base Actual - La Mesa', 'Base Actual - Otay', 'Base Actual - El Florido'],
        'lat': [32.533, 32.515, 32.528, 32.463],
        'lon': [-117.03, -116.98, -116.94, -116.82],
        'tipo': ['Actual'] * 4
    })
    
    # Bases optimizadas
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
---
*Autora:*
**M.C. Noelia Araceli Torres Cortés**

*Institución:*
**Tecnológico Nacional de México / Instituto Tecnológico de Tijuana**

*Directores de Tesis:*
- Dra. Yazmin Maldonado Robles
- Dr. Leonardo Trujillo Reyes
""")

pagina = st.sidebar.radio("Ir a:", 
    ["Resumen de la Tesis", "Datos y Corrección de Tiempo", "Clustering de Demanda", "Optimización de Ubicaciones", "Evolución del Sistema con IA Avanzada"]
)

st.sidebar.info("Los datos son simulados para fines de demostración, reflejando los conceptos y la geografía de la investigación original.")


# --- Renderizado de la Página ---

if pagina == "Resumen de la Tesis":
    st.title("Sistema de Despacho para Ambulancias de la Ciudad de Tijuana")
    st.subheader("Dashboard de la Tesis de Doctorado por la M.C. Noelia Araceli Torres Cortés")
    st.markdown("""
    Este dashboard proporciona un resumen interactivo de la investigación doctoral destinada a optimizar los Servicios Médicos de Emergencia (SME) para la Cruz Roja en Tijuana, México (CRT). El proyecto aborda el desafío crítico de reducir los tiempos de respuesta de las ambulancias en una ciudad con recursos limitados y condiciones urbanas complejas.
    """)
    with st.expander("**Fundamento Matemático del Problema General**"):
        st.markdown(r"""
        Desde una perspectiva matemática, el problema central abordado en esta tesis es un **problema de optimización combinatoria** conocido como un **Problema de Localización-Asignación** (*Location-Allocation Problem*), específicamente formulado como un **Problema de Cobertura** (*Covering Problem*).

        El objetivo general es determinar la ubicación óptima de un conjunto de $P$ ambulancias (recursos) en un conjunto de $J$ posibles localizaciones para maximizar la cobertura de un conjunto de $I$ puntos de demanda, sujeto a restricciones de tiempo de respuesta y recursos.

        La formulación matemática busca optimizar una función objetivo, $f(Y)$, donde $Y$ es el conjunto de decisiones de localización, para maximizar una métrica de efectividad del sistema (ej. el número de llamadas cubiertas). La novedad de esta tesis radica en la **calibración de los parámetros del modelo** (específicamente los tiempos de viaje $t_{ij}$) mediante técnicas de aprendizaje automático, uniendo así dos dominios matemáticos: la **investigación de operaciones** y la **estadística computacional**.
        """)
    st.header("Contribución Principal y Novedad")
    # ... (El resto del contenido de la página sigue igual) ...
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Modelo de Corrección de Tiempo de Viaje**")
        st.write("""
        La innovación principal es un modelo de aprendizaje automático que corrige las estimaciones de tiempo de viaje de las API estándar (como OSRM). Aprende la discrepancia entre las predicciones de la API y los tiempos de viaje reales de las ambulancias, teniendo en cuenta factores como el uso de la sirena y las exenciones a las leyes de tránsito. Esto resultó en una **mejora del 20% en la cobertura de ubicación**.
        """)

    with col2:
        st.info("🌐 **Aplicación en el Mundo Real**")
        st.write("""
        A diferencia de los estudios en ciudades bien estructuradas, esta investigación aborda la realidad 'desordenada' de una región en desarrollo. Al crear una solución práctica y basada en datos para la Cruz Roja de Tijuana, cierra la brecha entre la teoría académica y el impacto en el terreno. El modelo final utiliza **OSRM**, una herramienta gratuita y de código abierto, haciéndolo una **solución sostenible para la organización**, que no puede subsidiar gastos en APIs comerciales.
        """)

elif pagina == "Datos y Corrección de Tiempo":
    st.title("Exploración de Datos y Corrección del Tiempo de Viaje")
    st.markdown("Un desafío fundamental en la optimización del despacho de ambulancias es predecir con precisión cuánto tiempo tardará una ambulancia en llegar a un incidente.")
    st.subheader("Mapa de Llamadas de Emergencia Simuladas en Tijuana")
    st.map(df_llamadas[['lat', 'lon']], zoom=11, use_container_width=True)
    st.subheader("Corrección de las Estimaciones de Tiempo de Viaje")

    with st.expander("**Fundamento Matemático del Modelo de Corrección**"):
        st.markdown(r"""
        La decisión clave de la tesis es transformar el problema de **regresión** (predecir un valor continuo de tiempo de viaje) en un problema de **clasificación**. Matemáticamente, esta es una decisión estratégica para aumentar la robustez del modelo.

        1.  **Definición del Error:** Sea $T_{\text{real}}$ el tiempo de viaje real y $T_{\text{API}}$ la estimación del API. El error, $\epsilon$, se define como:
        """)
        st.latex(r''' \epsilon = T_{\text{API}} - T_{\text{real}} ''')
        st.markdown(r"""
        Un modelo de regresión que intente predecir $T_{\text{real}}$ directamente sería muy sensible a valores atípicos (viajes inusualmente rápidos o lentos), los cuales son comunes en el transporte de emergencia.

        2.  **Discretización del Espacio de Error:** El problema se simplifica al discretizar el espacio de error en un conjunto de clases, $C = \{c_1, c_2, c_3\}$, donde cada clase representa un rango de error (ej. $c_1: \epsilon > 7 \text{ min}$, $c_2: 2 < \epsilon \le 7 \text{ min}$, etc.).

        3.  **El Clasificador como Función:** Se entrena una función de clasificación $f: \mathcal{X} \to C$, donde $\mathcal{X}$ es el espacio de características (features) del viaje. La tesis utiliza un **Random Forest**, un método de ensamble que construye múltiples árboles de decisión y promedia sus resultados. Matemáticamente, esto reduce la varianza del modelo en comparación con un solo árbol, haciéndolo menos propenso al sobreajuste (overfitting).
        """)
        st.latex(r''' \hat{c} = f(X) ''')
        st.markdown(r"""
        4.  **Aplicación de la Corrección:** Una vez que se predice la clase $\hat{c}$ para un nuevo viaje, se aplica una corrección de tiempo precalculada, $\Delta_{\hat{c}}$. La tesis utiliza la mediana del error para cada clase, una elección estadísticamente robusta:
        """)
        st.latex(r''' \Delta_{\hat{c}} = \text{median}(\{\epsilon_i \mid \text{clase}(\epsilon_i) = \hat{c}\}) ''')
        st.markdown(r"""
        El tiempo de viaje final y corregido, $T_{\text{corregido}}$, se convierte en:
        """)
        st.latex(r''' T_{\text{corregido}} = T_{\text{API}} - \Delta_{\hat{c}} ''')
        st.markdown(r"""
        **Significado:** Este enfoque sacrifica una granularidad potencialmente engañosa por una **predicción categórica más confiable y estable**. En lugar de predecir un tiempo exacto con alta incertidumbre, el sistema predice un "tipo de viaje" con mayor certeza y aplica una corrección robusta y representativa para ese tipo.
        """)
    # ... (El resto del contenido visual de la página sigue aquí) ...
    error_antes = df_llamadas['tiempo_api_minutos'] - df_llamadas['tiempo_real_minutos']
    error_despues = df_llamadas['tiempo_corregido_minutos'] - df_llamadas['tiempo_real_minutos']
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Antes de la Corrección** (API vs. Tiempo Real)")
        fig1 = px.histogram(error_antes, nbins=50, title="Distribución del Error (API - Real)")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.markdown("**Después de la Corrección** (Modelo ML vs. Tiempo Real)")
        fig2 = px.histogram(error_despues, nbins=50, title="Distribución del Error (Corregido - Real)")
        st.plotly_chart(fig2, use_container_width=True)

elif pagina == "Clustering de Demanda":
    st.title("Identificación de Puntos de Alta Demanda Mediante Clustering")
    st.markdown("Para determinar dónde ubicar las ambulancias, las llamadas de emergencia históricas se agruparon utilizando K-Means. El centro de cada clúster representa un 'punto de demanda' o una zona de alta concentración.")
    
    with st.expander("**Fundamento Matemático del Clustering K-Means**"):
        st.markdown(r"""
        El algoritmo K-Means es un método de optimización no supervisado. Su objetivo es particionar un conjunto de $n$ observaciones (ubicaciones de llamadas de emergencia) $\{x_1, x_2, \dots, x_n\}$ en $k$ conjuntos o clústeres $S = \{S_1, S_2, \dots, S_k\}$ para minimizar la inercia o la **suma de cuadrados dentro del clúster** (WCSS - Within-Cluster Sum of Squares).

        La **función objetivo** que K-Means minimiza es:
        """)
        st.latex(r''' \arg\min_{S} \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2 ''')
        st.markdown(r"""
        Donde:
        - $x$ es una observación individual (un vector de coordenadas $[lat, lon]$).
        - $S_i$ es el conjunto de todas las observaciones en el clúster $i$.
        - $\mu_i$ es el **centroide** (la media de los puntos) del clúster $S_i$.
        - $\|x - \mu_i\|^2$ es la distancia Euclidiana al cuadrado entre un punto y el centroide de su clúster.

        **Significado:** La solución de este problema de minimización nos proporciona los centroides $\mu_1, \dots, \mu_k$. Estos centroides no son simplemente promedios; son los **centros de masa de la demanda de emergencias**. Transforman una nube de datos ruidosa y de alta cardinalidad (cientos de llamadas) en un conjunto pequeño y manejable de $k$ puntos de demanda representativos. Esta **reducción de dimensionalidad** del espacio del problema es un paso computacionalmente esencial antes de abordar el problema de optimización de ubicación, que es mucho más complejo.
        """)
    
    k = st.slider("Seleccione el Número de Clústeres de Demanda (k):", min_value=5, max_value=25, value=15, step=1)
    # ... (El resto del código de la página sigue aquí) ...
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    df_llamadas['cluster'] = kmeans.fit_predict(df_llamadas[['lat', 'lon']])
    centroides = kmeans.cluster_centers_
    df_centroides = pd.DataFrame(centroides, columns=['lat', 'lon'])
    st.subheader(f"Mapa de {k} Clústeres de Llamadas de Emergencia")
    fig = px.scatter_map(
        df_llamadas, lat="lat", lon="lon", color="cluster", zoom=10, height=600,
        title="Llamadas de Emergencia Coloreadas por Clúster"
    )
    fig.add_scattermapbox(
        lat=df_centroides['lat'], lon=df_centroides['lon'], mode='markers',
        marker=dict(size=15, symbol='star', color='red'), name='Punto de Alta Demanda'
    )
    st.plotly_chart(fig, use_container_width=True)

elif pagina == "Optimización de Ubicaciones":
    st.title("Optimización de la Ubicación de Ambulancias")
    st.markdown("Utilizando los puntos de alta demanda y los tiempos de viaje corregidos, se utilizó el Modelo Robusto de Doble Estándar (RDSM) para encontrar las ubicaciones óptimas para las ambulancias con el fin de maximizar la cobertura en toda la ciudad.")
    
    with st.expander("**Fundamento Matemático del Modelo de Optimización (RDSM)**"):
        st.markdown(r"""
        El RDSM es un modelo de **programación lineal entera binaria**. Su objetivo es tomar decisiones discretas (ubicar o no una ambulancia en un sitio) para maximizar una función objetivo bajo un conjunto de restricciones lineales.

        **Componentes del Modelo:**
        - **Conjuntos:**
            - $I$: Conjunto de puntos de demanda (los centroides de K-Means).
            - $J$: Conjunto de ubicaciones candidatas para las ambulancias.
        - **Parámetros:**
            - $P$: Número total de ambulancias disponibles.
            - $w_i$: Peso del punto de demanda $i$ (ej. número de llamadas en el clúster).
            - $t_{ij}$: Tiempo de viaje **corregido por ML** desde la ubicación $j$ al punto de demanda $i$.
            - $T_{\text{crit}}$: Umbral de tiempo de respuesta crítico.
        - **Variables de Decisión:**
            - $y_j \in \{0, 1\}$: $1$ si se ubica una base en el sitio $j$, $0$ en caso contrario.
            - $z_i \in \{0, 1\}$: $1$ si el punto de demanda $i$ está cubierto por al menos **dos** ambulancias, $0$ en caso contrario.

        **Formulación Matemática:**
        
        **Función Objetivo (Maximizar Doble Cobertura Ponderada):**
        """)
        st.latex(r''' \text{Maximizar} \quad Z = \sum_{i \in I} w_i z_i ''')
        st.markdown(r""" **Sujeto a las siguientes restricciones:** """)
        st.latex(r''' \sum_{j \in J \text{ s.t. } t_{ij} \le T_{\text{crit}}} y_j \ge 2z_i \quad \forall i \in I ''')
        st.markdown(r"""
        (1. Restricción de Doble Cobertura: Para que un punto $i$ se considere doblemente cubierto ($z_i=1$), la suma de ambulancias ($y_j$) que pueden llegar a él en el tiempo $T_{\text{crit}}$ debe ser al menos 2).
        """)
        st.latex(r''' \sum_{j \in J} y_j \le P ''')
        st.markdown(r"""
        (2. Restricción de Presupuesto: El número total de bases de ambulancias no puede exceder el número disponible $P$).

        **Significado:** Esta formulación matemática transforma un complejo problema logístico en un problema bien definido que puede ser resuelto por solvers de optimización. La solución, $\{y_j^*\}$, representa la **configuración de bases de ambulancias probadamente óptima** dadas las entradas. La integración del tiempo corregido $t_{ij}$ es lo que ancla este modelo teórico a la realidad operacional de Tijuana, y el enfoque en la **doble cobertura** añade una capa de robustez y resiliencia al sistema.
        """)
    # ... (El resto del contenido visual de la página sigue aquí) ...
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Resultados de la Optimización")
        st.metric(label="Cobertura Doble (Antes de Corrección)", value="80.0%")
        st.metric(label="Cobertura Doble (Después de Corrección)", value="100%", delta="20.0%")
    with col2:
        st.subheader("Ubicaciones de Ambulancias: Optimizadas vs. Actuales")
        todas_las_bases = pd.concat([bases_actuales, bases_optimizadas], ignore_index=True)
        fig = px.scatter_map(
            todas_las_bases, lat="lat", lon="lon", color="tipo",
            color_discrete_map={ "Actual": "orange", "Optimizada": "green" }
        )
        st.plotly_chart(fig, use_container_width=True)

elif pagina == "Evolución del Sistema con IA Avanzada":
    st.title("🚀 Evolución del Sistema con IA de Vanguardia")
    st.markdown("""
    Actuando como un SME líder en Machine Learning, esta sección describe una hoja de ruta estratégica para evolucionar el robusto sistema actual. El objetivo es pasar de un modelo de soporte a decisiones estático a un sistema dinámico, predictivo y adaptativo en tiempo real, utilizando bibliotecas de IA de código abierto de última generación.
    """)
    # ... (El contenido de esta página se mantiene igual que en la versión anterior) ...
    st.header("1. Predicción de Tiempos de Viaje de Próxima Generación")
    st.markdown("El modelo **Random Forest** actual es un excelente punto de partida. La siguiente evolución se centraría en capturar relaciones más complejas y dinámicas del tráfico.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Propuesta de Mejora: Modelos de Gradient Boosting**")
        st.write("Reemplazar o aumentar el Random Forest con modelos como **XGBoost, LightGBM, o CatBoost**. Estos algoritmos suelen ofrecer una mayor precisión en datos tabulares.")
        st.code("import xgboost as xgb\nimport lightgbm as lgb", language="python")

    with col2:
        st.info("🧠 **Visión a Futuro: Redes Neuronales Gráficas (GNNs)**")
        st.write("Modelar la red de calles de Tijuana como un grafo. Las GNNs pueden aprender las características del tráfico y los tiempos de viaje directamente de la topología de la ciudad.")
        st.code("import torch_geometric", language="python")

    st.header("2. Modelado de Demanda Dinámico y Predictivo")
    st.markdown("El clustering con **K-Means** sobre datos históricos es efectivo para identificar centros de demanda estáticos. El siguiente paso es predecir la demanda *antes* de que ocurra.")

    col1, col2 = st.columns(2)
    with col1:
        st.info("📈 **Propuesta de Mejora: Pronóstico Espacio-Temporal**")
        st.write("Utilizar modelos de series de tiempo para **predecir la probabilidad de llamadas de emergencia** por zona y por hora, permitiendo una reubicación proactiva de ambulancias.")
        st.code("from prophet import Prophet", language="python")
    
    with col2:
        st.info("🤖 **Visión a Futuro: Digital Twin (Gemelo Digital) y Simulación**")
        st.write("Crear una simulación de alta fidelidad del sistema de emergencias de Tijuana usando bibliotecas como **SimPy** para probar miles de escenarios hipotéticos y encontrar la estrategia más **robusta**.")
        st.code("import simpy", language="python")

    st.header("3. Hacia un Despacho y Reubicación Autónoma en Tiempo Real")
    st.info("🚀 **Propuesta de Vanguardia: Aprendizaje por Refuerzo (Reinforcement Learning)**")
    st.write("""
    Formular el problema de despacho como un entorno de RL, donde un **Agente** (el sistema de despacho) aprende una política óptima para realizar **Acciones** (enviar/reubicar ambulancias) basado en el **Estado** del sistema para maximizar una **Recompensa** (minimizar tiempos de respuesta).
    """)
    st.code("from stable_baselines3 import PPO", language="python")
