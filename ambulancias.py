import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Dashboard de Optimización de Despacho de Ambulancias",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data Caching and Generation ---
@st.cache_data
def cargar_datos():
    """Genera datos simulados realistas restringidos al área de Tijuana, México."""
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

# --- Sidebar Navigation ---
st.sidebar.title("🚑 Navegación")
st.sidebar.markdown("""
Este dashboard presenta los hallazgos clave de la tesis de doctorado:
**"Sistema de despacho para ambulancias de la ciudad de Tijuana"**
---
*Autora:*
**M.C. Noelia Araceli Torres Cortés**
""")
pagina = st.sidebar.radio("Ir a:", 
    ["Resumen de la Tesis", "Datos y Corrección de Tiempo", "Clustering de Demanda", "Optimización de Ubicaciones", "Evolución del Sistema con IA Avanzada"]
)

# --- Page Rendering ---
# The content for the first four pages remains unchanged.
if pagina == "Resumen de la Tesis":
    st.title("Resumen de la Tesis")
    # ... content ...

elif pagina == "Datos y Corrección de Tiempo":
    st.title("Datos y Corrección de Tiempo")
    # ... content ...

elif pagina == "Clustering de Demanda":
    st.title("Clustering de Demanda")
    # ... content ...
    k = st.slider("Seleccione el Número de Clústeres de Demanda (k):", min_value=5, max_value=25, value=15, step=1)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    df_llamadas['cluster'] = kmeans.fit_predict(df_llamadas[['lat', 'lon']])
    # ... rest of the page ...

elif pagina == "Optimización de Ubicaciones":
    st.title("Optimización de Ubicaciones")
    # ... content ...

elif pagina == "Evolución del Sistema con IA Avanzada":
    st.title("🚀 Evolución del Sistema con IA de Vanguardia")
    st.markdown("Esta sección interactiva, diseñada por un SME en Machine Learning, demuestra cómo las bibliotecas de IA de código abierto pueden evolucionar el sistema actual hacia una plataforma de despacho predictiva y adaptativa en tiempo real.")

    tab1, tab2, tab3 = st.tabs(["**1. Modelos Predictivos Superiores**", "**2. Predicción de Demanda**", "**3. Simulación y RL**"])

    with tab1:
        st.header("Mejora del Modelo de Corrección de Tiempo")
        st.markdown("El modelo **Random Forest** de la tesis es robusto. Sin embargo, los modelos de **Gradient Boosting** como **XGBoost** y **LightGBM** son el estándar actual en la industria para datos tabulares, ya que a menudo ofrecen mayor precisión.")

        @st.cache_data
        def train_and_compare_models():
            # **Just-in-Time Import:** Libraries are imported inside the function.
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
            with st.spinner("Entrenando modelos... (Esto puede tardar un momento la primera vez)"):
                results = train_and_compare_models()
                st.success("¡Modelos entrenados!")
                df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']).reset_index()
                fig = px.bar(df_results, x='index', y='Accuracy', title='Comparación de Precisión de Modelos', text_auto='.2%')
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Pronóstico Espacio-Temporal de la Demanda")
        st.markdown("En lugar de basar la ubicación en la demanda histórica, podemos usar modelos como **Prophet** de Meta para predecir dónde y cuándo *ocurrirán* las llamadas.")

        days_to_forecast = st.slider("Días a pronosticar en el futuro:", 7, 90, 30)

        @st.cache_data
        def generate_forecast(days):
            # **Just-in-Time Import:**
            from prophet import Prophet
            
            start_date = "2022-01-01"
            periods = 365
            df = pd.DataFrame({'ds': pd.date_range(start_date, periods=periods)})
            df['day_of_week'] = df['ds'].dt.dayofweek
            noise = np.random.randn(periods) * 5
            df['y'] = 100 + df['day_of_week'] * 5 + np.sin(df.index/30) * 10 + noise
            df['y'] = df['y'].astype(int)
            
            model = Prophet()
            model.fit(df)
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
        st.markdown("Para entrenar un agente de **Aprendizaje por Refuerzo (RL)**, primero necesitamos un entorno simulado. **SimPy** nos permite crear un 'gemelo digital' de la operación de despacho para probar políticas en un entorno sin riesgos.")
        
        num_ambulances = st.slider("Número de ambulancias en el sistema:", 1, 10, 3)
        avg_call_interval = st.slider("Tiempo promedio entre llamadas (minutos):", 5, 60, 15)

        @st.cache_data
        def run_simulation(ambulances, interval):
            # **Just-in-Time Import:**
            import simpy
            import random
            
            wait_times = []
            def call_process(env, call_id, ambulance_fleet):
                call_arrival_time = env.now
                with ambulance_fleet.request() as request:
                    yield request
                    wait_times.append(env.now - call_arrival_time)
                    yield env.timeout(random.uniform(15, 30))

            def call_generator(env, ambulance_fleet):
                for i in range(200): # Simulate a fixed number of calls
                    env.process(call_process(env, i, ambulance_fleet))
                    yield env.timeout(random.expovariate(1.0 / interval))

            env = simpy.Environment()
            ambulance_fleet = simpy.Resource(env, capacity=ambulances)
            env.process(call_generator(env, ambulance_fleet))
            env.run()
            
            return np.mean(wait_times) if wait_times else 0

        if st.button("🔬 Ejecutar Simulación de Despacho"):
            with st.spinner("Simulando eventos..."):
                avg_wait = run_simulation(num_ambulances, avg_call_interval)
                st.success("¡Simulación completada!")
                st.metric("Tiempo Promedio de Espera por una Ambulancia", f"{avg_wait:.2f} minutos")
                st.markdown("Un agente de RL sería entrenado para tomar decisiones que minimicen esta métrica.")
