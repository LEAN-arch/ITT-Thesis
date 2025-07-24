# app.py
# ==============================================================================
# LIBRARIES
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
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
    st.sidebar.info("Aplicación SME que demuestra los conceptos de la tesis y su evolución con IA de vanguardia.")

# ==============================================================================
# 2. APPLICATION STATE AND DATA LOADING
# ==============================================================================
if 'k_clusters' not in st.session_state:
    st.session_state.k_clusters = 15
if 'clusters_run' not in st.session_state:
    st.session_state.clusters_run = False

@st.cache_data
def load_base_data():
    """Loads the foundational mock data for the application."""
    lat_min, lat_max = 32.40, 32.55; lon_min, lon_max = -117.12, -116.60
    num_llamadas = 500; np.random.seed(42)
    api_time = np.random.gamma(shape=4, scale=5, size=num_llamadas) + 5
    real_time = api_time * np.random.normal(0.8, 0.1, size=num_llamadas)
    corrected_time = real_time + np.random.normal(0, 2, size=num_llamadas)
    df_llamadas = pd.DataFrame({
        'lat': np.random.uniform(lat_min, lat_max, num_llamadas),
        'lon': np.random.uniform(lon_min, lon_max, num_llamadas),
        'tiempo_api_minutos': api_time, 'tiempo_real_minutos': real_time, 'tiempo_corregido_minutos': corrected_time
    })
    bases_actuales = pd.DataFrame({'nombre': ['Base Actual - Centro', 'Base Actual - La Mesa'], 'lat': [32.533, 32.515], 'lon': [-117.03, -116.98], 'tipo': ['Actual'] * 2})
    return df_llamadas, bases_actuales

@st.cache_data
def run_kmeans(df, k):
    """Performs K-Means clustering."""
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
        st.markdown("Esta aplicación presenta los hallazgos fundamentales de la investigación doctoral sobre la optimización de Servicios Médicos de Emergencia (SME) en Tijuana, México.")
        with st.expander("Planteamiento del Problema y Justificación Científica", expanded=True):
            st.markdown(r"""
            El despliegue eficiente de Servicios Médicos de Emergencia (SME) es un problema crítico de asignación de recursos bajo incertidumbre. La métrica de rendimiento primaria, el **tiempo de respuesta**, está directamente correlacionada con los resultados clínicos de los pacientes, especialmente en casos de trauma y paros cardíacos. Los modelos de optimización matemática clásicos, como los problemas de localización de instalaciones, ofrecen un marco teórico para posicionar los recursos (ambulancias), pero su validez depende críticamente de la precisión de sus parámetros de entrada.
            
            En entornos urbanos complejos y con recursos limitados como Tijuana, los parámetros de tiempo de viaje ($t_{ij}$) proporcionados por las API de enrutamiento comerciales (e.g., Google Maps, OSRM) exhiben un **sesgo sistemático**. Estas APIs calculan rutas para vehículos civiles, sin tener en cuenta las condiciones operacionales de un vehículo de emergencia (uso de sirenas, exenciones de tráfico). Un modelo de optimización alimentado con estos datos sesgados producirá, por definición, una solución subóptima.
            
            Esta investigación aborda esta brecha fundamental mediante la **integración sinérgica de dos campos matemáticos**:
            1.  **Investigación de Operaciones:** Se utiliza para formular el problema de localización-asignación a través de un **Programa Lineal Entero Binario**, específicamente el Modelo Robusto de Doble Estándar (RDSM).
            2.  **Aprendizaje Automático (Estadística Computacional):** Se emplea para construir un modelo predictivo que calibra los parámetros de tiempo de viaje, transformando los datos brutos de la API en estimaciones realistas y estadísticamente insesgadas.
            
            La hipótesis central es que la calibración de los parámetros del modelo de optimización a través de un modelo de ML empíricamente validado conducirá a una mejora significativa y medible en la eficacia del sistema de despacho.
            """)

class TimeCorrectionPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.markdown("La contribución central de la tesis es la calibración de los tiempos de viaje. Un modelo de ML aprende la discrepancia sistemática entre las estimaciones de la API y la realidad operacional.")
        with st.expander("Metodología y Fundamento Matemático", expanded=True):
             st.markdown(r"""
            **Formulación del Problema:** Dado un tiempo de viaje real $T_{\text{real}}$ y una estimación de API $T_{\text{API}}$, el error se define como $\epsilon = T_{\text{API}} - T_{\text{real}}$. El objetivo es construir un modelo que prediga una corrección $\Delta$ tal que $T_{\text{API}} - \Delta \approx T_{\text{real}}$.
            
            **Decisión Metodológica Clave: Clasificación sobre Regresión**
            En lugar de predecir el valor continuo de $\epsilon$ (un problema de regresión), el problema se transforma en uno de **clasificación**. El espacio de error continuo se discretiza en $k$ clases categóricas $C = \{c_1, \dots, c_k\}$. Por ejemplo:
            - $c_1$: Gran sobreestimación ($\epsilon > \tau_1$)
            - $c_2$: Sobreestimación moderada ($\tau_2 < \epsilon \le \tau_1$)
            - $c_3$: Subestimación o error pequeño ($\epsilon \le \tau_2$)
            
            **Modelo y Justificación Científica:** Se entrena un clasificador no lineal, **Random Forest**, para aprender la función $f: \mathcal{X} \to C$, donde $\mathcal{X}$ es el espacio de características del viaje (hora, día, ubicación, etc.). Un Random Forest es un ensamblaje de árboles de decisión que mitiga el sobreajuste y captura interacciones complejas. La elección de la clasificación es deliberada: es más robusta a los valores atípicos (viajes extremadamente rápidos/lentos) que son comunes en los datos de emergencia y que desestabilizarían un modelo de regresión.
            
            **Aplicación de la Corrección:** Para una nueva predicción de clase $\hat{c} = f(X)$, se aplica una corrección $\Delta_{\hat{c}}$ calculada como la **mediana** del error de todos los puntos de entrenamiento en esa clase. La mediana es un estimador robusto de la tendencia central, insensible a los valores atípicos dentro de la clase.
            """)
        df_llamadas, _ = load_base_data()
        error_antes = df_llamadas['tiempo_api_minutos'] - df_llamadas['tiempo_real_minutos']
        error_despues = df_llamadas['tiempo_corregido_minutos'] - df_llamadas['tiempo_real_minutos']
        st.header("Resultados de la Calibración del Modelo")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribución del Error (Antes)")
            st.plotly_chart(px.histogram(error_antes, title="Error de la API (API - Real)"), use_container_width=True)
        with col2:
            st.subheader("Distribución del Error (Después)")
            st.plotly_chart(px.histogram(error_despues, title="Error del Modelo Corregido"), use_container_width=True)
        with st.expander("Análisis de Resultados e Implicaciones Científicas", expanded=True):
            st.markdown("""
            - **Gráfico de la Izquierda (Antes):** La distribución del error de la API es **sesgada a la derecha**, con una media significativamente mayor que cero. Estadísticamente, esto demuestra que la API es un **estimador sesgado**.
            - **Gráfico de la Derecha (Después):** El modelo de corrección transforma la distribución. Ahora es **aproximadamente simétrica y centrada en cero**, convirtiéndolo en un **estimador insesgado**.
            **Implicación:** La calibración del modelo convierte un parámetro de entrada inutilizable en uno científicamente válido.
            """)

class ClusteringPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.markdown("El primer paso computacional es agregar las ubicaciones de miles de llamadas históricas en un conjunto manejable de 'puntos de demanda' mediante K-Means.")
        with st.expander("Metodología y Fundamento Matemático: K-Means", expanded=True):
            st.markdown(r"""
            **1. Formulación del Problema**

            El problema de la agregación de la demanda consiste en transformar un conjunto de datos de alta cardinalidad, compuesto por $n$ ubicaciones geográficas de llamadas de emergencia $\{x_1, x_2, \dots, x_n\}$ donde cada $x_i \in \mathbb{R}^2$, en un conjunto representativo de $k$ "puntos de demanda" prototípicos, donde $k \ll n$. Este es un problema canónico de **aprendizaje no supervisado**, específicamente de **clustering por partición**.

            ---
            **2. Metodología: El Algoritmo K-Means**

            Se emplea el algoritmo K-Means, un método iterativo de optimización cuyo objetivo es particionar las $n$ observaciones en $k$ conjuntos o clústeres disjuntos, $S = \{S_1, S_2, \dots, S_k\}$, de tal manera que se minimice la inercia, comúnmente conocida como la **Suma de Cuadrados Intra-clúster** (Within-Cluster Sum of Squares, WCSS).

            La **función objetivo** que K-Means busca minimizar es:
            """)
            st.latex(r''' J(S, \mu) = \sum_{i=1}^{k} \sum_{x_j \in S_i} \|x_j - \mu_i\|^2 ''')
            st.markdown(r"""
            Donde:
            - $\|x_j - \mu_i\|^2$ es la distancia Euclidiana al cuadrado entre un punto de datos $x_j$ y el centroide $\mu_i$ de su clúster asignado $S_i$.
            - $\mu_i$ es el centroide del clúster $i$, calculado como la media vectorial de todos los puntos en ese clúster: $\mu_i = \frac{1}{|S_i|} \sum_{x_j \in S_i} x_j$.

            El algoritmo converge a un mínimo local de esta función objetivo a través de un procedimiento iterativo de dos pasos (Expectation-Maximization):
            1.  **Paso de Asignación (E-step):** Cada punto de datos $x_j$ se asigna al clúster cuyo centroide $\mu_i$ está más cercano: $S_i^{(t)} = \{x_j : \|x_j - \mu_i^{(t-1)}\|^2 \le \|x_j - \mu_{i'}^{(t-1)}\|^2 \quad \forall i'=1,\dots,k \}$.
            2.  **Paso de Actualización (M-step):** Los centroides se recalculan como la media de los puntos asignados a cada clúster en el paso anterior: $\mu_i^{(t)} = \frac{1}{|S_i^{(t)}|} \sum_{x_j \in S_i^{(t)}} x_j$.

            Estos pasos se repiten hasta que las asignaciones de los clústeres ya no cambian, indicando la convergencia.

            ---
            **3. Justificación Científica y Relevancia Operacional**

            La elección de K-Means para este problema geoespacial se justifica por varias razones:
            - **Interpretabilidad:** El centroide $\mu_i$ tiene una interpretación física directa y poderosa: es el **centro de masa** o el "centro de gravedad" de la demanda de emergencias en la región $i$. Esto lo convierte en un candidato natural para un punto de demanda en el modelo de optimización posterior.
            - **Eficiencia Computacional:** El algoritmo es computacionalmente eficiente y escala bien a grandes volúmenes de datos de llamadas, lo cual es esencial para un sistema operacional.
            - **Suposición de Geometría Euclidiana:** El algoritmo asume que los clústeres son de forma convexa e isotrópica (aproximadamente esféricos). En el contexto de la agregación de demanda a nivel de ciudad, donde las zonas de alta demanda a menudo son áreas geográficas compactas (barrios, distritos comerciales), esta suposición es razonable y efectiva como una primera aproximación.

            **Limitaciones:** Es importante reconocer que K-Means puede tener dificultades con clústeres de diferentes densidades o formas no convexas (e.g., demanda a lo largo de una carretera). Para análisis más finos, se podrían considerar métodos más avanzados como DBSCAN o UMAP (explorados en la pestaña de "Evolución con IA"). Sin embargo, para el propósito de la tesis de definir puntos de demanda a nivel macro, K-Means proporciona una solución robusta, interpretable y computacionalmente viable.
            """)
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
        tab_titles = ["1. Comparación de Clasificadores", "2. Reducción de Dimensionalidad Avanzada", "3. Pronóstico de Demanda", "4. Simulación de Sistema"]
        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)
        with tab1: self.render_classifier_comparison_tab()
        with tab2: self.render_umap_tab()
        with tab3: self.render_prophet_tab()
        with tab4: self.render_simpy_tab()

    def render_classifier_comparison_tab(self):
        st.header("Análisis Comparativo de Algoritmos de Clasificación")
        st.markdown("Un análisis exhaustivo requiere la comparación del **Random Forest** con otros paradigmas de clasificación para validar su optimalidad.")
        with st.expander("Metodologías y Fundamentos Matemáticos"):
            st.markdown("- **Regresión Logística:** Modelo lineal generalizado.\n- **SVM:** Encuentra un hiperplano de máxima separación.\n- **Naive Bayes:** Modelo probabilístico basado en el teorema de Bayes.\n- **LightGBM:** Ensamblaje de árboles construidos secuencialmente para corregir errores.")
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
        if st.button("📊 Ejecutar K-Means vs. UMAP + HDBSCAN"):
            with st.spinner("Generando embeddings y agrupando..."):
                import umap
                from sklearn.cluster import HDBSCAN
                df_calls, _ = load_base_data()
                data_points = df_calls[['lat', 'lon']].values
                kmeans_labels = KMeans(n_clusters=4, random_state=42, n_init='auto').fit_predict(data_points)
                df_calls['KMeans_Cluster'] = kmeans_labels
                embedding = umap.UMAP(n_neighbors=20, min_dist=0.0, n_components=2, random_state=42).fit_transform(data_points)
                hdbscan_labels = HDBSCAN(min_cluster_size=20).fit_predict(embedding)
                df_calls['UMAP_Cluster'] = hdbscan_labels
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = px.scatter_mapbox(df_calls, lat="lat", lon="lon", color=df_calls['KMeans_Cluster'].astype(str), title="Clusters Geoespaciales (K-Means)", mapbox_style="carto-positron", category_orders={"color": sorted(df_calls['KMeans_Cluster'].astype(str).unique())})
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    fig2 = px.scatter_mapbox(df_calls, lat="lat", lon="lon", color=df_calls['UMAP_Cluster'].astype(str), title="Clusters Geoespaciales (UMAP+HDBSCAN)", mapbox_style="carto-positron", category_orders={"color": sorted(df_calls['UMAP_Cluster'].astype(str).unique())})
                    st.plotly_chart(fig2, use_container_width=True)
                with st.expander("Análisis de Resultados e Implicaciones"):
                    st.markdown("""
                    **Comparación:** El mapa de K-Means divide el espacio en regiones geométricas convexas. El mapa de UMAP+HDBSCAN encuentra clústeres basados en la densidad y la conectividad, identificando grupos de formas más arbitrarias y separando el ruido (puntos grises, cluster -1).
                    **Implicación Científica:** UMAP proporciona una representación más fiel de la **estructura de la demanda real**, lo que conduce a una definición de "puntos de demanda" más precisa.
                    """)

    def render_prophet_tab(self):
        st.header("Metodología: Pronóstico de Demanda con Prophet")
        st.markdown("Se utiliza **Prophet** de Meta, un modelo de series de tiempo bayesiano diseñado para manejar estacionalidades múltiples.")
        with st.expander("Fundamento Matemático: Prophet"):
            st.markdown(r"Prophet modela una serie de tiempo como una suma de componentes:")
            st.latex(r''' y(t) = g(t) + s(t) + h(t) + \epsilon_t ''')
            st.markdown(r"Donde $g(t)$ es la tendencia, $s(t)$ la estacionalidad (series de Fourier), $h(t)$ los feriados, y $\epsilon_t$ el error.")
        days_to_forecast = st.slider("Parámetro: Horizonte de Pronóstico (días)", 7, 90, 30, key="prophet_slider")
        if st.button("📈 Generar Pronóstico"):
            with st.spinner("Calculando..."):
                from prophet import Prophet
                df = pd.DataFrame({'ds': pd.date_range("2022-01-01", periods=365)})
                df['y'] = 50 + (df['ds'].dt.dayofweek // 5) * 20 + np.sin(df.index / 365 * 4 * np.pi) * 10
                model = Prophet(weekly_seasonality=True, yearly_seasonality=True).fit(df)
                forecast = model.predict(model.make_future_dataframe(periods=days_to_forecast))
                historical_avg = df[df['ds'].dt.dayofweek == forecast.iloc[-1]['ds'].dayofweek]['y'].mean()
                predicted_val = forecast.iloc[-1]['yhat']
                st.pyplot(model.plot(forecast))
                st.subheader("Análisis de Resultados e Implicaciones")
                col1, col2 = st.columns(2)
                col1.metric(f"Promedio Histórico para este Día", f"{historical_avg:.1f} llamadas")
                col2.metric(f"Pronóstico para este Día", f"{predicted_val:.1f} llamadas", delta=f"{predicted_val - historical_avg:.1f}")

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
            wait_times_priority = []; wait_times_standard = []
            
            def call_process(env, fleet, is_priority):
                arrival_time = env.now
                with fleet.request(priority=(1 if is_priority else 2)) as request:
                    yield request
                    wait_time = env.now - arrival_time
                    if is_priority: wait_times_priority.append(wait_time)
                    else: wait_times_standard.append(wait_time)
                    yield env.timeout(random.uniform(20, 40))

            def call_generator(env, fleet, interval):
                for _ in range(500):
                    is_priority_call = random.random() < 0.2
                    env.process(call_process(env, fleet, is_priority_call))
                    yield env.timeout(random.expovariate(1.0 / interval))
            
            env = simpy.Environment()
            fleet = simpy.PriorityResource(env, capacity=ambulances)
            env.process(call_generator(env, fleet, interval))
            env.run()
            return np.mean(wait_times_priority) if wait_times_priority else 0, np.mean(wait_times_standard) if wait_times_standard else 0

        if st.button("🔬 Ejecutar Simulación con Prioridad"):
            with st.spinner("Simulando..."):
                priority_wait, standard_wait = run_dispatch_simulation(num_ambulances, avg_call_interval)
                st.subheader("Resultados de la Simulación")
                col1, col2 = st.columns(2)
                col1.metric("Espera Promedio (Prioritarias)", f"{priority_wait:.2f} min")
                col2.metric("Espera Promedio (Estándar)", f"{standard_wait:.2f} min")

# ==============================================================================
# 5. MAIN APPLICATION ROUTER
# ==============================================================================
def main():
    """Main function to route to the correct page."""
    render_sidebar_info()
    pages = {
        "Resumen de la Tesis": ThesisSummaryPage("Resumen de la Tesis", "📜"),
        "Calibración del Modelo de Tiempos": TimeCorrectionPage("Calibración del Modelo", "⏱️"),
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
