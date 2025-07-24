# ==============================================================================
# LIBRARIES
# Import lightweight libraries at the top for fast startup.
# Heavy libraries are imported "just-in-time" inside their respective functions.
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Dashboard de Optimización de Despacho de Ambulancias",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# DATA LOADING AND CACHING
# ==============================================================================
@st.cache_data
def cargar_datos():
    """
    Genera datos simulados realistas restringidos al área de Tijuana, México.
    This function is cached, so it only runs once.
    """
    lat_min, lat_max = 32.40, 32.55
    lon_min, lon_max = -117.12, -116.60
    num_llamadas = 500
    np.random.seed(42)
    latitudes = np.random.uniform(lat_min, lat_max, num_llamadas)
    longitudes = np.random.uniform(lon_min, lon_max, num_llamadas)
    tiempo_api = np.random.uniform(5, 30, num_llamadas)
    tiempo_real = tiempo_api * np.random.uniform(0.6, 0.95, num_llamadas)
    tiempo_corregido = tiempo_real * np.random.uniform(0.95, 1.05, num_llamadas)
    df_llamadas = pd.DataFrame({
        'lat': latitudes, 'lon': longitudes, 'tiempo_api_minutos': tiempo_api,
        'tiempo_real_minutos': tiempo_real, 'tiempo_corregido_minutos': tiempo_corregido
    })
    bases_actuales = pd.DataFrame({
        'nombre': ['Base Actual - Centro', 'Base Actual - La Mesa', 'Base Actual - Otay', 'Base Actual - El Florido'],
        'lat': [32.533, 32.515, 32.528, 32.463], 'lon': [-117.03, -116.98, -116.94, -116.82], 'tipo': ['Actual'] * 4
    })
    num_optimizadas = 12
    bases_optimizadas = pd.DataFrame({
        'nombre': [f'Estación Optimizada {i+1}' for i in range(num_optimizadas)],
        'lat': np.random.uniform(lat_min, lat_max, num_optimizadas),
        'lon': np.random.uniform(lon_min, lon_max, num_optimizadas), 'tipo': ['Optimizada'] * num_optimizadas
    })
    return df_llamadas, bases_actuales, bases_optimizadas

df_llamadas, bases_actuales, bases_optimizadas = cargar_datos()

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
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

PAGES = [
    "Resumen de la Tesis",
    "Datos y Corrección de Tiempo",
    "Clustering de Demanda",
    "Optimización de Ubicaciones",
    "Evolución del Sistema con IA Avanzada"
]

pagina = st.sidebar.radio("Ir a:", PAGES)
st.sidebar.info("Los datos son simulados para fines de demostración, reflejando los conceptos y la geografía de la investigación original.")

# ==============================================================================
# PAGE 1: RESUMEN DE LA TESIS
# ==============================================================================
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

# ==============================================================================
# PAGE 2: DATOS Y CORRECCIÓN DE TIEMPO
# ==============================================================================
elif pagina == "Datos y Corrección de Tiempo":
    st.title("Exploración de Datos y Corrección del Tiempo de Viaje")
    st.markdown("Un desafío fundamental en la optimización del despacho de ambulancias es predecir con precisión cuánto tiempo tardará una ambulancia en llegar a un incidente.")
    st.subheader("Mapa de Llamadas de Emergencia Simuladas en Tijuana")
    st.map(df_llamadas[['lat', 'lon']], zoom=11, use_container_width=True)
    st.subheader("Corrección de las Estimaciones de Tiempo de Viaje")

    with st.expander("**Fundamento Matemático del Modelo de Corrección**"):
        st.markdown(r"""
        La decisión clave de la tesis es transformar el problema de **regresión** (predecir un valor continuo) en **clasificación**. Esto aumenta la robustez del modelo frente a valores atípicos.
        """)
        st.latex(r''' \epsilon = T_{\text{API}} - T_{\text{real}} \quad \rightarrow \quad \hat{c} = f(X) \quad \rightarrow \quad T_{\text{corregido}} = T_{\text{API}} - \Delta_{\hat{c}} ''')
        st.markdown(r"Este enfoque predice una 'categoría de viaje' con mayor certeza y aplica una corrección robusta (mediana) para esa categoría.")

    error_antes = df_llamadas['tiempo_api_minutos'] - df_llamadas['tiempo_real_minutos']
    error_despues = df_llamadas['tiempo_corregido_minutos'] - df_llamadas['tiempo_real_minutos']
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Antes de la Corrección** (API vs. Tiempo Real)")
        fig1 = px.histogram(error_antes, nbins=50, title="Distribución del Error (API - Real)")
        fig1.update_layout(yaxis_title="Frecuencia", xaxis_title="Error de Tiempo (minutos)")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.markdown("**Después de la Corrección** (Modelo ML vs. Tiempo Real)")
        fig2 = px.histogram(error_despues, nbins=50, title="Distribución del Error (Corregido - Real)")
        fig2.update_layout(yaxis_title="Frecuencia", xaxis_title="Error de Tiempo (minutos)")
        st.plotly_chart(fig2, use_container_width=True)

# ==============================================================================
# PAGE 3: CLUSTERING DE DEMANDA
# ==============================================================================
elif pagina == "Clustering de Demanda":
    st.title("Identificación de Puntos de Alta Demanda Mediante Clustering")
    st.markdown("Para determinar dónde ubicar las ambulancias, las llamadas de emergencia históricas se agruparon utilizando K-Means. El centro de cada clúster representa un 'punto de demanda'.")
    with st.expander("**Fundamento Matemático del Clustering K-Means**"):
        st.markdown(r"K-Means particiona $n$ observaciones en $k$ clústeres al minimizar la suma de cuadrados dentro del clúster (WCSS):")
        st.latex(r''' \arg\min_{S} \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2 ''')
        st.markdown(r"Los centroides resultantes, $\mu_i$, son los **centros de masa de la demanda**, reduciendo la complejidad del problema para la etapa de optimización.")

    k = st.slider("Seleccione el Número de Clústeres de Demanda (k):", min_value=5, max_value=25, value=15, step=1)
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

# ==============================================================================
# PAGE 4: OPTIMIZACIÓN DE UBICACIONES
# ==============================================================================
elif pagina == "Optimización de Ubicaciones":
    st.title("Optimización de la Ubicación de Ambulancias")
    st.markdown("Utilizando los puntos de alta demanda y los tiempos de viaje corregidos, se utilizó el Modelo Robusto de Doble Estándar (RDSM) para encontrar las ubicaciones óptimas.")
    
    with st.expander("**Fundamento Matemático del Modelo de Optimización (RDSM)**"):
        st.markdown(r"El RDSM es un modelo de **programación lineal entera binaria** que busca maximizar la cobertura doble ponderada.")
        st.latex(r''' \text{Maximizar} \quad Z = \sum_{i \in I} w_i z_i \quad \text{sujeto a} \quad \sum_{j \in J, t_{ij} \le T_{\text{crit}}} y_j \ge 2z_i, \quad \sum y_j \le P ''')
        st.markdown(r"La solución $\{y_j^*\}$ representa la configuración de bases **probablemente óptima**, anclada a la realidad operacional mediante los tiempos de viaje corregidos por ML.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Resultados de la Optimización")
        st.metric(label="Cobertura Doble (Antes de Corrección)", value="80.0%", help="Cobertura usando tiempos de API estándar.")
        st.metric(label="Cobertura Doble (Después de Corrección)", value="100%", delta="20.0%", help="Cobertura usando tiempos corregidos por ML, demostrando el impacto del proyecto.")
    with col2:
        st.subheader("Ubicaciones de Ambulancias: Optimizadas vs. Actuales")
        todas_las_bases = pd.concat([bases_actuales, bases_optimizadas], ignore_index=True)
        fig = px.scatter_map(
            todas_las_bases, lat="lat", lon="lon", color="tipo",
            title="Comparación de Ubicaciones de Bases",
            hover_name="nombre",
            color_discrete_map={"Actual": "orange", "Optimizada": "green"}
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 5: EVOLUCIÓN DEL SISTEMA CON IA AVANZADA (INTERACTIVE DEMOS)
# ==============================================================================
elif pagina == "Evolución del Sistema con IA Avanzada":
    st.title("🚀 Evolución del Sistema con IA de Vanguardia")
    st.markdown("Esta sección interactiva demuestra cómo las bibliotecas de IA de código abierto pueden evolucionar el sistema actual hacia una plataforma de despacho predictiva y adaptativa en tiempo real.")

    tab1, tab2, tab3 = st.tabs(["**1. Modelos Predictivos Superiores**", "**2. Predicción de Demanda**", "**3. Simulación y RL**"])

    with tab1:
        st.header("Mejora del Modelo de Corrección de Tiempo")
        st.markdown("El modelo **Random Forest** de la tesis es robusto. Sin embargo, los modelos de **Gradient Boosting** como **XGBoost** y **LightGBM** son el estándar actual en la industria para datos tabulares, ya que a menudo ofrecen mayor precisión.")

        @st.cache_data
        def train_and_compare_models():
            """Trains and compares RandomForest, XGBoost, and LightGBM models."""
            import xgboost as xgb
            import lightgbm as lgb
            X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, n_classes=3, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
            rf_acc = accuracy_score(y_test, rf.predict(X_test))
            xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss').fit(X_train, y_train)
            xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
            lgb_model = lgb.LGBMClassifier(random_state=42).fit(X_train, y_train)
            lgb_acc = accuracy_score(y_test, lgb_model.predict(X_test))
            return {"Random Forest": rf_acc, "XGBoost": xgb_acc, "LightGBM": lgb_acc}

        if st.button("▶️ Entrenar y Comparar Modelos Predictivos"):
            with st.spinner("Entrenando modelos... (Puede tardar la primera vez)"):
                results = train_and_compare_models()
                st.success("¡Modelos entrenados!")
                df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']).reset_index()
                fig = px.bar(df_results, x='index', y='Accuracy', title='Comparación de Precisión', text_auto='.2%')
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Pronóstico Espacio-Temporal de la Demanda")
        st.markdown("En lugar de basar la ubicación en la demanda histórica, podemos usar modelos como **Prophet** para predecir dónde y cuándo *ocurrirán* las llamadas.")
        days_to_forecast = st.slider("Días a pronosticar en el futuro:", 7, 90, 30)

        @st.cache_data
        def generate_forecast(days):
            """Generates a time series forecast using Prophet."""
            from prophet import Prophet
            start_date = "2022-01-01"
            periods = 365
            df = pd.DataFrame({'ds': pd.date_range(start_date, periods=periods)})
            df['y'] = 100 + (df['ds'].dt.dayofweek // 5) * 25 + np.sin(df.index / 365 * 2 * np.pi) * 10 + np.random.randn(periods) * 5
            model = Prophet(weekly_seasonality=True, yearly_seasonality=True).fit(df)
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            return model, forecast

        if st.button("📈 Generar Pronóstico de Demanda"):
            with st.spinner(f"Calculando pronóstico... (La primera ejecución de Prophet puede ser lenta)"):
                model, forecast = generate_forecast(days_to_forecast)
                st.success("¡Pronóstico generado!")
                fig = model.plot(forecast)
                st.pyplot(fig)

    with tab3:
        st.header("Simulación de Sistema con SimPy")
        st.markdown("Para entrenar un agente de **Aprendizaje por Refuerzo (RL)**, primero necesitamos un entorno simulado. **SimPy** nos permite crear un 'gemelo digital' de la operación de despacho.")
        num_ambulances = st.slider("Número de ambulancias en el sistema:", 1, 10, 3)
        avg_call_interval = st.slider("Tiempo promedio entre llamadas (minutos):", 5, 60, 15)

        @st.cache_data
        def run_simulation(ambulances, interval, num_calls_to_sim):
            """Runs a discrete-event simulation of an ambulance dispatch system."""
            import simpy
            import random
            wait_times = []
            def call_process(env, ambulance_fleet):
                call_arrival_time = env.now
                with ambulance_fleet.request() as request:
                    yield request
                    wait_times.append(env.now - call_arrival_time)
                    yield env.timeout(random.uniform(20, 45))
            def call_generator(env, ambulance_fleet, interval):
                for i in range(num_calls_to_sim):
                    env.process(call_process(env, ambulance_fleet))
                    yield env.timeout(random.expovariate(1.0 / interval))
            env = simpy.Environment()
            ambulance_fleet = simpy.Resource(env, capacity=ambulances)
            env.process(call_generator(env, ambulance_fleet, interval))
            env.run()
            return np.mean(wait_times) if wait_times else 0

        if st.button("🔬 Ejecutar Simulación de Despacho"):
            with st.spinner("Simulando cientos de llamadas y despachos..."):
                avg_wait = run_simulation(num_ambulances, avg_call_interval, num_calls_to_sim=500)
                st.success("¡Simulación completada!")
                st.metric("Tiempo Promedio de Espera por una Ambulancia", f"{avg_wait:.2f} minutos")
                st.markdown("Un agente de **Aprendizaje por Refuerzo** sería entrenado para tomar decisiones que minimicen esta métrica.")
