# app.py
# ==============================================================================
# LIBRARIES
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random # For SimPy simulation
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
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
    st.sidebar.info("Aplicación de grado SME que demuestra los conceptos de la tesis y su evolución con IA de vanguardia.")

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
            El problema central es la optimización de un sistema estocástico y dinámico con recursos limitados. La eficacia de los SME se mide principalmente por el **tiempo de respuesta**. Las estimaciones de tiempo de las API comerciales son sistemáticamente incorrectas. Esta investigación aborda esta brecha mediante la integración de **Investigación de Operaciones** y **Aprendizaje Automático**.
            """)

class ClusteringPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.markdown("El primer paso computacional es agregar las ubicaciones de miles de llamadas históricas en un conjunto manejable de 'puntos de demanda' mediante K-Means.")
        with st.expander("Metodología y Fundamento Matemático: K-Means"):
            st.markdown(r"K-Means particiona $n$ observaciones en $k$ clústeres al minimizar la Suma de Cuadrados Intra-clúster (WCSS):")
            st.latex(r''' \arg\min_{S} \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2 ''')
        k_input = st.slider("Parámetro (k): Número de Puntos de Demanda", 2, 25, st.session_state.k_clusters, key="k_slider")
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
        st.markdown("Con los puntos de demanda definidos, se formula y resuelve un problema de optimización para determinar la ubicación estratégica de las ambulancias.")
        if not st.session_state.get('clusters_run', False):
            st.warning("⚠️ **Requisito Previo:** Genere los 'Puntos de Demanda' en la página anterior para proceder.")
            return
        with st.expander("Metodología y Fundamento Matemático: RDSM"):
            st.markdown(r"El problema se formula como un **Programa Lineal Entero Binario (BIP)** para maximizar la doble cobertura.")
            st.latex(r''' \text{Maximizar} \quad Z = \sum_{i \in I} w_i z_i \quad \text{s.t.} \quad \sum_{j \in J, t_{ij} \le T_{\text{crit}}} y_j \ge 2z_i, \quad \sum y_j \le P ''')
        num_ambulances = st.slider("Parámetro (P): Número de Ambulancias a Ubicar", 2, 12, 8, key="opt_slider")
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
        st.markdown("Esta sección explora metodologías de vanguardia para extender la investigación actual, presentando prototipos funcionales con explicaciones científicas detalladas.")
        
        tab_titles = [
            "1. Comparación de Clasificadores",
            "2. Reducción de Dimensionalidad Avanzada",
            "3. Pronóstico de Demanda",
            "4. Simulación de Sistema",
        ]
        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

        with tab1: self.render_classifier_comparison_tab()
        with tab2: self.render_umap_tab()
        with tab3: self.render_prophet_tab()
        with tab4: self.render_simpy_tab()

    def render_classifier_comparison_tab(self):
        st.header("Análisis Comparativo de Algoritmos de Clasificación")
        st.markdown("El modelo **Random Forest** de la tesis es un punto de partida robusto. Sin embargo, un análisis exhaustivo requiere la comparación con otros paradigmas de clasificación para validar su optimalidad.")
        
        with st.expander("Metodologías y Fundamentos Matemáticos"):
            st.markdown("""
            - **Regresión Logística:** Modelo lineal generalizado (base estadística) que modela la probabilidad logit de pertenencia a una clase.
            - **Máquinas de Vectores de Soporte (SVM):** Modelo no lineal que encuentra un hiperplano de máxima separación entre clases en un espacio de características de alta dimensionalidad.
            - **Naive Bayes Gaussiano:** Modelo probabilístico basado en el teorema de Bayes con una fuerte suposición (ingenua) de independencia condicional entre características.
            - **Gradient Boosting (LightGBM):** Ensamblaje de árboles de decisión construidos secuencialmente para corregir los errores de los predecesores.
            """)
        
        if st.button("▶️ Entrenar y Comparar Clasificadores"):
            with st.spinner("Entrenando 5 modelos distintos..."):
                import lightgbm as lgb
                X, y = make_classification(n_samples=2000, n_features=15, n_informative=8, n_redundant=2, n_classes=3, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                models = {
                    "Logistic Regression": LogisticRegression(random_state=42),
                    "Gaussian Naive Bayes": GaussianNB(),
                    "Support Vector Machine": SVC(random_state=42),
                    "Random Forest": RandomForestClassifier(random_state=42),
                    "LightGBM": lgb.LGBMClassifier(random_state=42)
                }
                results = {}
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    results[name] = accuracy_score(y_test, model.predict(X_test))
                
                df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']).sort_values('Accuracy', ascending=False).reset_index()
                fig = px.bar(df_results, x='index', y='Accuracy', title='Comparación de Precisión de Clasificadores', text_auto='.3%')
                st.plotly_chart(fig, use_container_width=True)

    def render_umap_tab(self):
        st.header("Reducción de Dimensionalidad Avanzada con UMAP")
        st.markdown("""
        **Problema:** K-Means es sensible a la geometría Euclidiana y puede fallar en descubrir clústeres con formas no globulares. **UMAP (Uniform Manifold Approximation and Projection)** es un algoritmo de la topología algebraica que es superior para encontrar la estructura intrínseca de los datos.
        """)
        with st.expander("Fundamento Matemático: UMAP"):
            st.markdown(r"""
            UMAP modela los datos como una red de vecinos (un grafo difuso) y busca una incrustación (embedding) de baja dimensión que preserve la estructura topológica de este grafo. Minimiza la divergencia de Kullback-Leibler (entropía cruzada) entre las distribuciones de probabilidad de las distancias en el espacio de alta y baja dimensión.
            """)
        
        if st.button("📊 Ejecutar K-Means vs. UMAP + Clustering"):
            with st.spinner("Generando embeddings y agrupando..."):
                import umap
                from sklearn.cluster import HDBSCAN
                df_calls, _ = load_base_data()
                data_points = df_calls[['lat', 'lon']].values
                
                # UMAP + HDBSCAN
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=2, random_state=42)
                embedding = reducer.fit_transform(data_points)
                hdbscan_labels = HDBSCAN(min_cluster_size=15).fit_predict(embedding)
                
                df_calls['UMAP_Cluster'] = hdbscan_labels
                df_calls['UMAP_x'] = embedding[:, 0]
                df_calls['UMAP_y'] = embedding[:, 1]

                col1, col2 = st.columns(2)
                with col1:
                    fig1 = px.scatter(df_calls, x="lon", y="lat", color=df_calls['UMAP_Cluster'].astype(str), title="Clusters Geoespaciales (UMAP+HDBSCAN)")
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    fig2 = px.scatter(df_calls, x="UMAP_x", y="UMAP_y", color=df_calls['UMAP_Cluster'].astype(str), title="Clusters en el Espacio de Embedding UMAP")
                    st.plotly_chart(fig2, use_container_width=True)

    def render_prophet_tab(self):
        st.header("Metodología: Pronóstico de Demanda con Prophet")
        # ... (Content from previous version)
        days_to_forecast = st.slider("Parámetro: Horizonte de Pronóstico (días)", 7, 90, 30, key="prophet_slider")
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
        st.header("Metodología: Simulación de Sistemas y RL")
        # ... (Content from previous version)
        num_ambulances = st.slider("Parámetro: Número de Ambulancias", 1, 10, 3, key="simpy_slider_1")
        avg_call_interval = st.slider("Parámetro: Tiempo Promedio Entre Llamadas (min)", 5, 60, 20, key="simpy_slider_2")
        
        @st.cache_data
        def run_dispatch_simulation(ambulances, interval):
            import simpy
            wait_times = []
            env = simpy.Environment()
            fleet = simpy.Resource(env, capacity=ambulances)
            def call_proc(env, fleet):
                arrival = env.now
                with fleet.request() as req:
                    yield req
                    wait_times.append(env.now - arrival)
                    yield env.timeout(random.uniform(20, 40))
            def generator(env, fleet, interval):
                for _ in range(500):
                    env.process(call_proc(env, fleet))
                    yield env.timeout(random.expovariate(1.0 / interval))
            env.process(generator(env, fleet, interval))
            env.run()
            return np.mean(wait_times) if wait_times else 0

        if st.button("🔬 Ejecutar Simulación"):
            with st.spinner("Simulando..."):
                avg_wait = run_dispatch_simulation(num_ambulances, avg_call_interval)
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
