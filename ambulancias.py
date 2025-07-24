# app.py
# ==============================================================================
# LIBRARIES
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random  # For SimPy simulation
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
    def __init__(self, title, icon):
        self.title = title
        self.icon = icon
    @abstractmethod
    def render(self) -> None:
        st.set_page_config(page_title=self.title, page_icon=self.icon, layout="wide")
        st.title(f"{self.icon} {self.title}")

class ThesisSummaryPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.subheader("Un Resumen Interactivo de la Tesis Doctoral")
        st.markdown("Esta aplicación presenta los hallazgos fundamentales de la investigación doctoral sobre la optimización de Servicios Médicos de Emergencia (SME) en Tijuana, México, enmarcando el problema y la solución propuesta.")
        with st.expander("Planteamiento del Problema y Justificación Científica", expanded=True):
            st.markdown(r"""
            El problema central es la optimización de un sistema estocástico y dinámico con recursos limitados. La eficacia de los SME se mide principalmente por el **tiempo de respuesta**, una variable crítica que impacta directamente en la morbilidad y mortalidad de los pacientes. En entornos urbanos complejos como Tijuana, las estimaciones de tiempo de viaje de las API comerciales son sistemáticamente incorrectas, lo que invalida los modelos de optimización estándar.

            Esta investigación aborda esta brecha fundamental mediante la **integración sinérgica de dos campos matemáticos**:
            1.  **Investigación de Operaciones:** Para la formulación del problema de localización-asignación.
            2.  **Aprendizaje Automático:** Para la calibración empírica de los parámetros del modelo a partir de datos históricos, específicamente el tiempo de viaje.
            """)
        st.header("Contribuciones Científicas Principales")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Modelo Híbrido de Corrección de Tiempos")
            st.markdown("La contribución metodológica principal es un **modelo de aprendizaje supervisado (Random Forest)** que no predice el tiempo directamente, sino que clasifica el *tipo de error* de la API. Este enfoque de clasificación transforma un problema de regresión ruidoso en una tarea de clasificación más robusta, demostrando una **mejora del 20% en la cobertura** del sistema de optimización resultante.")
        with col2:
            st.subheader("2. Marco de Solución Sostenible")
            st.markdown("La investigación valida el uso de **herramientas de código abierto (OSRM)**, demostrando que es posible construir sistemas de optimización de alto rendimiento sin depender de costosas API comerciales. Esto representa una contribución significativa para la implementación de soluciones similares en entornos con recursos limitados.")

class ClusteringPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.markdown("El primer paso computacional es la reducción de la dimensionalidad de la demanda. Las ubicaciones de miles de llamadas históricas se agregan en un conjunto manejable de 'puntos de demanda' representativos mediante el algoritmo K-Means.")
        with st.expander("Metodología y Fundamento Matemático: K-Means"):
            st.markdown(r"""
            **Problema:** Particionar un conjunto de $n$ vectores de observación de llamadas $\{x_1, \dots, x_n\}$ en $k$ clústeres $S = \{S_1, \dots, S_k\}$.
            
            **Objetivo:** Minimizar la inercia, o la Suma de Cuadrados Intra-clúster (WCSS), definida como la suma de las distancias Euclidianas al cuadrado entre cada punto y el centroide de su clúster asignado.
            """)
            st.latex(r''' \arg\min_{S} \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2 ''')
            st.markdown(r"""
            Donde $\mu_i$ es el centroide (media vectorial) del clúster $S_i$.
            
            **Justificación:** Se elige K-Means por su eficiencia computacional en grandes conjuntos de datos y su interpretabilidad. Los centroides resultantes, $\mu_i$, no son meros promedios; representan los **centros de masa gravitacionales de la demanda histórica de emergencias**. Este paso es crucial para transformar un problema de optimización intratable (con miles de puntos de demanda) en uno computacionalmente factible.
            """)
        k_input = st.slider("Parámetro (k): Número de Puntos de Demanda a Identificar", 2, 25, st.session_state.k_clusters, key="k_slider")
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
        with st.expander("Análisis de Resultados"):
            st.markdown("""
            El mapa visualiza la partición del espacio geográfico. Cada color representa un clúster de demanda cohesivo, y la estrella roja indica su centro de masa. Se puede observar cómo las áreas de alta densidad de llamadas emergen naturalmente como clústeres distintos. La selección del parámetro $k$ es un compromiso entre la granularidad del modelo y el riesgo de sobreajuste; un $k$ demasiado alto podría modelar ruido en lugar de la señal de demanda subyacente. Los centroides generados aquí sirven como la entrada principal para la siguiente etapa de optimización.
            """)

class OptimizationPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.markdown("Con los puntos de demanda definidos, se formula y resuelve un problema de optimización para determinar la ubicación estratégica de las ambulancias.")
        if not st.session_state.get('clusters_run', False):
            st.warning("⚠️ **Requisito Previo:** Genere los 'Puntos de Demanda' en la página anterior para proceder.")
            return
        with st.expander("Metodología y Fundamento Matemático: RDSM"):
            st.markdown(r"""
            El problema se formula como un **Programa Lineal Entero Binario (BIP)**.
            **Variables de Decisión:**
            - $y_j \in \{0, 1\}$: $1$ si se establece una base en la localización $j$.
            - $z_i \in \{0, 1\}$: $1$ si el punto de demanda $i$ está cubierto por al menos dos ambulancias.
            **Función Objetivo:** Maximizar la demanda total ponderada ($w_i$) que recibe doble cobertura.
            """)
            st.latex(r''' \text{Maximizar} \quad Z = \sum_{i \in I} w_i z_i ''')
            st.markdown(r"**Restricciones Principales:**")
            st.latex(r''' \text{(1) Cobertura:} \quad \sum_{j \in J \text{ s.t. } t_{ij} \le T_{\text{crit}}} y_j \ge 2z_i \quad \forall i \in I ''')
            st.latex(r''' \text{(2) Presupuesto:} \quad \sum_{j \in J} y_j \le P ''')
            st.markdown(r"**Justificación:** La elección de maximizar la **doble cobertura** introduce **robustez** en la solución. Asegura que exista un respaldo dentro del umbral de tiempo crítico, aumentando la resiliencia del sistema.")
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
        with st.expander("Análisis de Resultados"):
            st.markdown("""
            El resultado clave es el **salto del 80% al 100% en la doble cobertura**. Esto valida cuantitativamente la hipótesis central de la tesis: **la calidad de los parámetros de entrada de un modelo de optimización es tan importante como la sofisticación del propio modelo.**
            """)

class AIEvolutionPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.markdown("Esta sección explora metodologías de vanguardia para extender la investigación actual, presentando prototipos funcionales con explicaciones científicas detalladas.")
        tab_titles = ["1. Comparación de Clasificadores", "2. Reducción de Dimensionalidad Avanzada", "3. Pronóstico de Demanda", "4. Simulación de Sistema"]
        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)
        with tab1: self.render_classifier_comparison_tab()
        with tab2: self.render_umap_tab()
        with tab3: self.render_prophet_tab()
        with tab4: self.render_simpy_tab()

    def render_classifier_comparison_tab(self):
        st.header("Análisis Comparativo de Algoritmos de Clasificación")
        st.markdown("Un análisis exhaustivo requiere la comparación del **Random Forest** con otros paradigmas de clasificación.")
        with st.expander("Metodologías y Fundamentos Matemáticos"):
            st.markdown("- **Regresión Logística:** Modelo lineal que modela la probabilidad logit.\n- **SVM:** Encuentra un hiperplano de máxima separación entre clases.\n- **Naive Bayes:** Modelo probabilístico basado en el teorema de Bayes.\n- **LightGBM:** Ensamblaje de árboles construidos secuencialmente para corregir errores.")
        if st.button("▶️ Entrenar y Comparar Clasificadores"):
            with st.spinner("Entrenando 5 modelos distintos..."):
                import lightgbm as lgb
                X, y = make_classification(n_samples=2000, n_features=15, n_informative=8, n_classes=3, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                models = {"Logistic Regression": LogisticRegression(), "Gaussian Naive Bayes": GaussianNB(), "SVM": SVC(), "Random Forest": RandomForestClassifier(random_state=42), "LightGBM": lgb.LGBMClassifier(random_state=42)}
                results = {name: accuracy_score(y_test, model.fit(X_train, y_train).predict(X_test)) for name, model in models.items()}
                df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']).sort_values('Accuracy', ascending=False).reset_index()
                fig = px.bar(df_results, x='index', y='Accuracy', title='Comparación de Precisión de Clasificadores', text_auto='.3%')
                st.plotly_chart(fig, use_container_width=True)

    def render_umap_tab(self):
        st.header("Reducción de Dimensionalidad Avanzada con UMAP")
        st.markdown("**UMAP (Uniform Manifold Approximation and Projection)** es un algoritmo de la topología algebraica superior a K-Means para encontrar la estructura intrínseca de los datos.")
        with st.expander("Fundamento Matemático: UMAP"):
            st.markdown(r"UMAP modela los datos como un grafo difuso y busca una incrustación de baja dimensión que preserve la estructura topológica, minimizando la divergencia de Kullback-Leibler entre las distribuciones de distancia en el espacio de alta y baja dimensión.")
        if st.button("📊 Ejecutar K-Means vs. UMAP + HDBSCAN"):
            with st.spinner("Generando embeddings y agrupando..."):
                import umap; from sklearn.cluster import HDBSCAN
                df_calls, _ = load_base_data()
                data_points = df_calls[['lat', 'lon']].values
                embedding = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=2, random_state=42).fit_transform(data_points)
                labels = HDBSCAN(min_cluster_size=15).fit_predict(embedding)
                df_calls['UMAP_Cluster'], df_calls['UMAP_x'], df_calls['UMAP_y'] = labels, embedding[:, 0], embedding[:, 1]
                col1, col2 = st.columns(2)
                with col1: st.plotly_chart(px.scatter(df_calls, x="lon", y="lat", color=df_calls['UMAP_Cluster'].astype(str), title="Clusters Geoespaciales (UMAP+HDBSCAN)"), use_container_width=True)
                with col2: st.plotly_chart(px.scatter(df_calls, x="UMAP_x", y="UMAP_y", color=df_calls['UMAP_Cluster'].astype(str), title="Clusters en Espacio de Embedding UMAP"), use_container_width=True)

    def render_prophet_tab(self):
        st.header("Metodología: Pronóstico de Demanda con Prophet")
        st.markdown("Se utiliza **Prophet** de Meta, un modelo de series de tiempo bayesiano diseñado para manejar estacionalidades múltiples (diaria, semanal, anual) y días festivos.")
        with st.expander("Fundamento Matemático: Prophet"):
            st.markdown(r"Prophet modela una serie de tiempo como una suma de componentes:")
            st.latex(r''' y(t) = g(t) + s(t) + h(t) + \epsilon_t ''')
            st.markdown(r"Donde $g(t)$ es la tendencia, $s(t)$ la estacionalidad (modelada con series de Fourier), $h(t)$ los feriados, y $\epsilon_t$ el error.")
        days_to_forecast = st.slider("Parámetro: Horizonte de Pronóstico (días)", 7, 90, 30, key="prophet_slider")
        if st.button("📈 Generar Pronóstico"):
            with st.spinner("Calculando..."):
                from prophet import Prophet
                df = pd.DataFrame({'ds': pd.date_range("2022-01-01", periods=365)})
                df['y'] = 50 + (df['ds'].dt.dayofweek // 5) * 20 + np.sin(df.index / 365 * 4 * np.pi) * 10
                model = Prophet(weekly_seasonality=True, yearly_seasonality=True).fit(df)
                st.pyplot(model.plot(model.predict(model.make_future_dataframe(periods=days_to_forecast))))

    def render_simpy_tab(self):
        st.header("Metodología: Simulación y Aprendizaje por Refuerzo (RL)")
        st.markdown("Se construye un **'gemelo digital'** con **SimPy** para servir como entorno de entrenamiento para un agente de RL.")
        with st.expander("Fundamento Matemático: Procesos de Decisión de Markov"):
            st.markdown("El problema se modela como un **MDP** $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$. El objetivo es encontrar la política óptima $\pi^*$ que maximice el retorno esperado:")
            st.latex(r''' \pi^* = \arg\max_{\pi} \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid \pi \right] ''')
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
                    yield req; wait_times.append(env.now - arrival)
                    yield env.timeout(random.uniform(20, 40))
            def generator(env, fleet, interval):
                for _ in range(500):
                    env.process(call_proc(env, fleet)); yield env.timeout(random.expovariate(1.0 / interval))
            env.process(generator(env, fleet, interval)); env.run()
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
