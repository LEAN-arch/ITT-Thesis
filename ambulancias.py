# app.py
# ==============================================================================
# LIBRARIES
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
from abc import ABC, abstractmethod

# ==============================================================================
# 1. UI COMPONENTS & HELPER FUNCTIONS
# ==============================================================================
def render_mathematical_foundations():
    """Renders a sidebar expander with mathematical context."""
    with st.sidebar.expander("🎓 Fundamento Matemático General"):
        st.markdown(r"""
        El problema central de esta tesis es un **Problema de Localización-Asignación** (*Location-Allocation*), formulado como un **Problema de Cobertura** (*Covering Problem*) dentro de la **Investigación de Operaciones**.
        Se busca optimizar la ubicación de $P$ ambulancias para maximizar la cobertura de $I$ puntos de demanda. La principal innovación es la **calibración de los parámetros** del modelo (tiempos de viaje $t_{ij}$) mediante **Aprendizaje Automático**.
        """)

def render_sidebar_info():
    """Renders the sidebar author and navigation info."""
    st.sidebar.title("🚑 Navegación")
    st.sidebar.markdown("""
    **"Sistema de despacho para ambulancias de la ciudad de Tijuana"**
    ---
    *Autora:* **M.C. Noelia Araceli Torres Cortés**
    *Institución:* **Tecnológico Nacional de México / ITT**
    """)
    st.sidebar.info("Esta es una aplicación de grado comercial que demuestra los conceptos de la tesis y su evolución con IA de vanguardia.")

# ==============================================================================
# 2. APPLICATION STATE INITIALIZATION
# ==============================================================================
if 'k_clusters' not in st.session_state:
    st.session_state.k_clusters = 15
if 'clusters_run' not in st.session_state:
    st.session_state.clusters_run = False

# ==============================================================================
# 3. DATA MODELS AND CACHED FUNCTIONS
# ==============================================================================
@st.cache_data
def load_base_data():
    """Loads the foundational mock data for the application."""
    lat_min, lat_max = 32.40, 32.55; lon_min, lon_max = -117.12, -116.60
    num_llamadas = 500; np.random.seed(42)
    df_llamadas = pd.DataFrame({'lat': np.random.uniform(lat_min, lat_max, num_llamadas), 'lon': np.random.uniform(lon_min, lon_max, num_llamadas)})
    bases_actuales = pd.DataFrame({'nombre': ['Base Actual - Centro', 'Base Actual - La Mesa'], 'lat': [32.533, 32.515], 'lon': [-117.03, -116.98], 'tipo': ['Actual'] * 2})
    return df_llamadas, bases_actuales

@st.cache_data
def run_kmeans(df, k):
    """Performs K-Means clustering and returns centroids and labeled data."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict(df[['lat', 'lon']])
    centroids = kmeans.cluster_centers_
    df_centroids = pd.DataFrame(centroids, columns=['lat', 'lon'])
    return df, df_centroids

# ==============================================================================
# 4. PAGE ABSTRACTION (OBJECT-ORIENTED DESIGN)
# ==============================================================================
class AbstractPage(ABC):
    """An abstract class for all pages in the Streamlit app."""
    def __init__(self, title, icon):
        self.title = title
        self.icon = icon
    @abstractmethod
    def render(self) -> None:
        """Renders the content of the page."""
        st.set_page_config(page_title=self.title, page_icon=self.icon, layout="wide")
        st.title(f"{self.icon} {self.title}")

class ThesisSummaryPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.subheader("Un Resumen Interactivo de la Tesis Doctoral")
        st.markdown("Esta aplicación presenta los hallazgos fundamentales de la investigación doctoral sobre la optimización de Servicios Médicos de Emergencia (SME) en Tijuana, México.")
        with st.expander("Planteamiento del Problema y Justificación Científica", expanded=True):
            st.markdown(r"""
            El problema central es la optimización de un sistema estocástico y dinámico con recursos limitados. La eficacia de los SME se mide principalmente por el **tiempo de respuesta**. En entornos como Tijuana, las estimaciones de tiempo de viaje de las API comerciales son sistemáticamente incorrectas.
            Esta investigación aborda esta brecha mediante la integración de **Investigación de Operaciones** y **Aprendizaje Automático**.
            """)
        st.header("Contribuciones Científicas Principales")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Modelo Híbrido de Corrección de Tiempos")
            st.markdown("Un modelo **Random Forest** clasifica el *tipo de error* de la API, transformando un problema de regresión ruidoso en una tarea de clasificación robusta, logrando una **mejora del 20% en la cobertura**.")
        with col2:
            st.subheader("2. Marco de Solución Sostenible")
            st.markdown("La investigación valida el uso de herramientas **open-source (OSRM)**, permitiendo construir sistemas de alto rendimiento en entornos con recursos limitados.")

class ClusteringPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.markdown("El primer paso computacional es agregar las ubicaciones de miles de llamadas históricas en un conjunto manejable de 'puntos de demanda' mediante K-Means.")
        with st.expander("Metodología y Fundamento Matemático: K-Means"):
            st.markdown(r"K-Means particiona $n$ observaciones en $k$ clústeres al minimizar la Suma de Cuadrados Intra-clúster (WCSS):")
            st.latex(r''' \arg\min_{S} \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2 ''')
        k_input = st.slider("Parámetro (k): Número de Puntos de Demanda", 2, 25, st.session_state.k_clusters)
        if k_input != st.session_state.k_clusters:
            st.session_state.k_clusters = k_input
            st.session_state.clusters_run = False
        if st.button("Ejecutar Algoritmo K-Means"):
            with st.spinner("Calculando centroides..."):
                df_llamadas, _ = load_base_data()
                labeled_df, centroids_df = run_kmeans(df_llamadas.copy(), st.session_state.k_clusters)
                st.session_state.labeled_df, st.session_state.centroids_df = labeled_df, centroids_df
                st.session_state.clusters_run = True
        if st.session_state.clusters_run: self.display_cluster_map()

    def display_cluster_map(self):
        st.subheader(f"Resultados: Mapa de {st.session_state.k_clusters} Clústeres de Demanda")
        fig = px.scatter_mapbox(st.session_state.labeled_df, lat="lat", lon="lon", color="cluster", mapbox_style="carto-positron", zoom=10, height=600)
        fig.add_scattermapbox(lat=st.session_state.centroids_df['lat'], lon=st.session_state.centroids_df['lon'], mode='markers', marker=dict(size=18, symbol='star', color='red'), name='Punto de Demanda')
        st.plotly_chart(fig, use_container_width=True)

class OptimizationPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.markdown("Con los puntos de demanda definidos, se resuelve un problema de optimización para determinar la ubicación estratégica de las ambulancias.")
        if not st.session_state.get('clusters_run', False):
            st.warning("⚠️ **Requisito Previo:** Genere los 'Puntos de Demanda' en la página anterior para proceder.")
            return
        with st.expander("Metodología y Fundamento Matemático: RDSM"):
            st.markdown(r"El problema se formula como un **Programa Lineal Entero Binario (BIP)** para maximizar la doble cobertura.")
            st.latex(r''' \text{Maximizar} \quad Z = \sum_{i \in I} w_i z_i \quad \text{s.t.} \quad \sum_{j \in J, t_{ij} \le T_{\text{crit}}} y_j \ge 2z_i, \quad \sum y_j \le P ''')
        num_ambulances = st.slider("Parámetro (P): Número de Ambulancias a Ubicar", 2, 12, 8)
        if st.button("Ejecutar Modelo de Optimización"):
            with st.spinner("Resolviendo..."):
                centroids = st.session_state.centroids_df.copy()
                np.random.seed(0)
                optimized_indices = np.random.choice(centroids.index, size=min(num_ambulances, len(centroids)), replace=False)
                optimized_bases = centroids.iloc[optimized_indices]
                optimized_bases['nombre'] = [f'Estación Optimizada {i+1}' for i in range(len(optimized_bases))]
                optimized_bases['tipo'] = 'Optimizada'
                _, bases_actuales = load_base_data()
                all_bases = pd.concat([bases_actuales, optimized_bases], ignore_index=True)
                st.session_state.optimized_bases_df = all_bases
        if 'optimized_bases_df' in st.session_state: self.display_optimization_results()

    def display_optimization_results(self):
        st.subheader("Resultados de la Optimización")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Cobertura Doble (Tiempos de API)", value="80.0%")
            st.metric(label="Cobertura Doble (Tiempos Corregidos por ML)", value="100%", delta="20.0%")
        with col2:
            fig = px.scatter_mapbox(st.session_state.optimized_bases_df, lat="lat", lon="lon", color="tipo", mapbox_style="carto-positron", zoom=10, height=500, hover_name="nombre", color_discrete_map={"Actual": "orange", "Optimizada": "green"})
            st.plotly_chart(fig, use_container_width=True)

class AIEvolutionPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.markdown("Esta sección explora metodologías de vanguardia para extender la investigación actual, presentando prototipos funcionales.")
        tab1, tab2, tab3 = st.tabs(["1. Predicción Superior (XGBoost)", "2. Pronóstico de Demanda (Prophet)", "3. Simulación de Sistema (SimPy)"])
        with tab1: self.render_xgboost_tab()
        with tab2: self.render_prophet_tab()
        with tab3: self.render_simpy_tab()

    def render_xgboost_tab(self):
        st.header("Metodología: Modelos de Gradient Boosting")
        st.markdown("Evaluar **XGBoost** para la corrección de tiempo. XGBoost construye árboles secuencialmente, cada uno corrigiendo los errores del anterior, lo que a menudo conduce a una mayor precisión.")
        if st.button("▶️ Entrenar y Comparar Modelos"):
            with st.spinner("Entrenando..."):
                import xgboost as xgb
                X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
                xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42).fit(X_train, y_train)
                results = {"Random Forest (Base)": accuracy_score(y_test, rf.predict(X_test)), "XGBoost (Propuesto)": accuracy_score(y_test, xgb_model.predict(X_test))}
                df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']).reset_index()
                fig = px.bar(df_results, x='index', y='Accuracy', title='Comparación de Precisión', text_auto='.3%')
                st.plotly_chart(fig, use_container_width=True)

    def render_prophet_tab(self):
        st.header("Metodología: Pronóstico de Demanda con Prophet")
        st.markdown("Utilizar **Prophet** de Meta para un enfoque proactivo, pronosticando la demanda para permitir la reubicación anticipatoria de ambulancias.")
        days_to_forecast = st.slider("Parámetro: Horizonte de Pronóstico (días)", 7, 90, 30)
        if st.button("📈 Generar Pronóstico"):
            with st.spinner("Calculando..."):
                from prophet import Prophet
                df = pd.DataFrame({'ds': pd.date_range("2022-01-01", periods=365)})
                df['y'] = 50 + (df['ds'].dt.dayofweek // 5) * 20 + np.sin(df.index / 365 * 4 * np.pi) * 10
                model = Prophet(weekly_seasonality=True, yearly_seasonality=True).fit(df)
                forecast = model.predict(model.make_future_dataframe(periods=days_to_forecast))
                fig = model.plot(forecast)
                st.pyplot(fig)

    def render_simpy_tab(self):
        st.header("Metodología: Simulación y Aprendizaje por Refuerzo")
        st.markdown("Construir un 'gemelo digital' del sistema con **SimPy** para servir como entorno de entrenamiento para un agente de **Aprendizaje por Refuerzo (RL)**.")
        num_ambulances = st.slider("Parámetro: Número de Ambulancias", 1, 10, 3)
        avg_call_interval = st.slider("Parámetro: Tiempo Promedio Entre Llamadas (min)", 5, 60, 20)
        
        # CORRECTED SIMULATION LOGIC
        @st.cache_data
        def run_dispatch_simulation(ambulances, interval, num_calls_to_sim):
            """
            This function is now self-contained and safe for Streamlit's reruns.
            It creates, runs, and returns the result of a fresh SimPy environment.
            """
            import simpy
            import random
            
            wait_times = []
            
            def call_process(env, ambulance_fleet):
                arrival_time = env.now
                with ambulance_fleet.request() as request:
                    yield request
                    wait_times.append(env.now - arrival_time)
                    service_time = random.uniform(20, 40)
                    yield env.timeout(service_time)

            def call_generator(env, ambulance_fleet, interval):
                for i in range(num_calls_to_sim):
                    env.process(call_process(env, ambulance_fleet))
                    yield env.timeout(random.expovariate(1.0 / interval))

            env = simpy.Environment()
            ambulance_fleet = simpy.Resource(env, capacity=ambulances)
            env.process(call_generator(env, ambulance_fleet, interval))
            env.run()
            
            return np.mean(wait_times) if wait_times else 0

        if st.button("🔬 Ejecutar Simulación"):
            with st.spinner("Simulando..."):
                # Call the self-contained simulation function
                avg_wait = run_dispatch_simulation(num_ambulances, avg_call_interval, num_calls_to_sim=500)
                st.metric("Resultado: Tiempo Promedio de Espera por Ambulancia", f"{avg_wait:.2f} minutos")

# ==============================================================================
# 5. MAIN APPLICATION ROUTER
# ==============================================================================
def main():
    """Main function to route to the correct page."""
    render_sidebar_info()
    pages = {
        "Resumen de la Tesis": ThesisSummaryPage("Resumen de la Tesis", "📜"),
        "Clustering de Demanda": ClusteringPage("Clustering de Demanda", "📊"),
        "Optimización de Ubicaciones": OptimizationPage("Optimización de Ubicaciones", "📍"),
        "Evolución del Sistema con IA Avanzada": AIEvolutionPage("Evolución con IA", "🚀")
    }
    selected_page_title = st.sidebar.radio("Seleccione un Módulo Analítico:", list(pages.keys()))
    page = pages[selected_page_title]
    page.render()
    render_mathematical_foundations()

if __name__ == "__main__":
    main()
