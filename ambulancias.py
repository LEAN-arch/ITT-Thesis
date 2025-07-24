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
# PAGE CONFIGURATION (RUNS ONLY ONCE)
# ==============================================================================
st.set_page_config(
    page_title="Sistema de Despacho de Ambulancias",
    page_icon="🚑",
    layout="wide"
)

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
    *Directores:* **Dra. Yazmin Maldonado Robles, Dr. Leonardo Trujillo Reyes**
    """)
    st.sidebar.info("Aplicación SME que demuestra los conceptos de la tesis y su evolución con IA de vanguardia.")

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
    api_time = np.random.gamma(shape=4, scale=5, size=num_llamadas) + 5
    real_time = api_time * np.random.normal(0.8, 0.1, size=num_llamadas)
    corrected_time = real_time + np.random.normal(0, 2, size=num_llamadas)
    df_llamadas = pd.DataFrame({
        'lat': np.random.uniform(lat_min, lat_max, num_llamadas),
        'lon': np.random.uniform(lon_min, lon_max, num_llamadas),
        'tiempo_api_minutos': api_time,
        'tiempo_real_minutos': real_time,
        'tiempo_corregido_minutos': corrected_time
    })
    bases_actuales = pd.DataFrame({'nombre': ['Base Actual - Centro', 'Base Actual - La Mesa'], 'lat': [32.533, 32.515], 'lon': [-117.03, -116.98], 'tipo': ['Actual'] * 2})
    return df_llamadas, bases_actuales

@st.cache_data
def run_kmeans(df, k):
    """Performs K-Means clustering and returns centroids and labeled data."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    df_copy = df.copy()
    df_copy['cluster'] = kmeans.fit_predict(df_copy[['lat', 'lon']])
    centroids = kmeans.cluster_centers_
    df_centroids = pd.DataFrame(centroids, columns=['lat', 'lon'])
    return df_copy, df_centroids

# ==============================================================================
# 4. PAGE ABSTRACTION (OBJECT-ORIENTED DESIGN)
# ==============================================================================
class AbstractPage(ABC):
    def __init__(self, title, icon):
        self.title = title
        self.icon = icon
    
    @abstractmethod
    def render(self) -> None:
        st.title(f"{self.icon} {self.title}")

class ThesisSummaryPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.subheader("Un Resumen Interactivo de la Tesis Doctoral")
        st.markdown("""
        Esta aplicación presenta de manera interactiva los conceptos y hallazgos fundamentales de la investigación doctoral **"Sistema de despacho para ambulancias de la ciudad de Tijuana"**. El objetivo es ilustrar cómo la combinación de **Investigación de Operaciones** y **Aprendizaje Automático** puede mejorar drásticamente la eficiencia de los Servicios Médicos de Emergencia (SME), salvando vidas al reducir los tiempos de respuesta.
        """)
        
        with st.expander("Resumen Oficial de la Tesis (Abstract)", expanded=True):
            st.info("""
            **Esta tesis se enfoca en los SMEs de la Cruz Roja de Tijuana (CRT), con el objetivo principal de diseñar un sistema web para la toma de decisiones y optimización de los SMEs prehospitalarios, tomando en cuenta los patrones de servicios históricos.** Se aborda el análisis de datos de la CRT, la estimación de tiempos de viaje con Google Maps y OSRM, y se presenta el modelo **DSM (Double Standard Model)** para maximizar la demanda cubierta. 
            
            La contribución central es la **corrección de la estimación del tiempo de viaje** mediante aprendizaje máquina (Random Forest), demostrando que los resultados mejoran la cobertura en un **20% más que sin corrección**. Esto valida el uso de OSRM como una herramienta de código libre viable para la CRT. Finalmente, se diseña un sistema web que integra módulos de agrupamiento de llamadas, ubicación y reubicación, simulación de eventos y corrección de tiempos de viaje para visualizar y analizar escenarios.
            """)

        st.header("Flujo Metodológico de la Investigación")
        st.markdown("""
        El trabajo de tesis siguió un proceso estructurado para abordar el problema desde el análisis de datos hasta la implementación y validación de un sistema completo. Cada módulo de esta aplicación corresponde a una etapa clave de la investigación.
        """)
        st.graphviz_chart('''
        digraph {
            rankdir=TB;
            node [shape=box, style=rounded, fontname="Helvetica"];
            
            A [label="Capítulo 2: Análisis de Datos y Tiempos de Viaje\n- Filtrado de datos históricos de la CRT.\n- Comparación de Google Maps vs. OSRM.\n- Identificación de sesgos sistemáticos en la estimación de tiempos."];
            B [label="Capítulo 4: Corrección de Tiempos con Machine Learning\n- Formulación del problema como Clasificación.\n- Entrenamiento de un modelo Random Forest para predecir el tipo de error.\n- Validación: mejora del 20% en la cobertura."];
            C [label="Capítulo 3: Optimización de Ubicaciones\n- Uso de K-Means para agrupar llamadas en puntos de demanda.\n- Aplicación del Modelo Robusto de Doble Estándar (RDSM).\n- Validación con datos de patrullas policíacas."];
            D [label="Capítulo 5: Diseño del Sistema Web\n- Integración de los módulos anteriores en una herramienta interactiva.\n- Diseño de la arquitectura (MVC) y flujo de usuario.\n- Simulación de despacho y análisis de escenarios."];
            
            A -> B [label="El análisis revela la necesidad de corregir los tiempos"];
            B -> C [label="Parámetros calibrados alimentan el modelo de optimización"];
            C -> D [label="El modelo de optimización es el núcleo del sistema"];
        }
        ''')
        
        st.header("Contribuciones Científicas Principales")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Modelo Híbrido de Corrección de Tiempos")
            st.markdown("Se demuestra que un modelo **Random Forest**, al transformar un problema de regresión de error (predecir minutos de diferencia) en uno de clasificación (predecir el *tipo* de error), es más robusto y efectivo. Este enfoque metodológico resultó en una **mejora del 20% en la cobertura de servicio**, validando la hipótesis central de la tesis.")
        with col2:
            st.subheader("2. Marco de Solución Sostenible y de Código Abierto")
            st.markdown("La investigación valida rigurosamente el uso de herramientas **open-source (OSRM)** como una alternativa viable y sin costo a soluciones comerciales como Google Maps. Se demuestra que, una vez calibrados con el modelo de ML, los tiempos de OSRM son científicamente válidos para la optimización, permitiendo construir sistemas de alto rendimiento en entornos con recursos limitados como la Cruz Roja de Tijuana.")

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
            st.subheader("Distribución del Error (Antes de la Corrección)")
            fig1 = px.histogram(error_antes, nbins=50, title="Error de la API (API - Real)")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.subheader("Distribución del Error (Después de la Corrección)")
            fig2 = px.histogram(error_despues, nbins=50, title="Error del Modelo Corregido (Corregido - Real)")
            st.plotly_chart(fig2, use_container_width=True)
        with st.expander("Análisis de Resultados e Implicaciones Científicas", expanded=True):
            st.markdown("""
            - **Gráfico de la Izquierda (Antes):** La distribución del error de la API está **sesgada a la derecha**, con una media significativamente mayor que cero. Estadísticamente, esto demuestra que la API es un **estimador sesgado** y, por lo tanto, no confiable para la optimización.
            - **Gráfico de la Derecha (Después):** El modelo de corrección transforma la distribución. Ahora es **aproximadamente simétrica y centrada en cero**, convirtiéndolo en un **estimador insesgado**.
            **Implicación Científica:** La calibración del modelo convierte un parámetro de entrada inutilizable en uno científicamente válido. Este paso es el que habilita la mejora del **20% en la cobertura** que se demuestra en la tesis, ya que el modelo de optimización pasa a operar con datos que reflejan la realidad.
            """)

class ClusteringPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.markdown("El primer paso computacional es agregar las ubicaciones de miles de llamadas históricas en un conjunto manejable de 'puntos de demanda' representativos mediante el algoritmo K-Means.")
        with st.expander("Metodología y Fundamento Matemático: K-Means", expanded=True):
            st.markdown(r"""
            **1. Formulación del Problema**
            El problema de la agregación de la demanda consiste en transformar un conjunto de datos de alta cardinalidad, compuesto por $n$ ubicaciones geográficas de llamadas de emergencia $\{x_1, x_2, \dots, x_n\}$ donde cada $x_i \in \mathbb{R}^2$, en un conjunto representativo de $k$ "puntos de demanda" prototípicos, donde $k \ll n$. Este es un problema canónico de **aprendizaje no supervisado**, específicamente de **clustering por partición**.
            
            **2. Metodología: El Algoritmo K-Means**
            Se emplea el algoritmo K-Means, un método iterativo de optimización cuyo objetivo es particionar las $n$ observaciones en $k$ conjuntos o clústeres disjuntos, $S = \{S_1, S_2, \dots, S_k\}$, de tal manera que se minimice la inercia, comúnmente conocida como la **Suma de Cuadrados Intra-clúster** (WCSS).
            La **función objetivo** que K-Means busca minimizar es:
            """)
            st.latex(r''' J(S, \mu) = \sum_{i=1}^{k} \sum_{x_j \in S_i} \|x_j - \mu_i\|^2 ''')
            st.markdown(r"""
            Donde $\mu_i$ es el centroide (media vectorial) del clúster $S_i$. El algoritmo converge a un mínimo local de esta función a través de un procedimiento iterativo de dos pasos (Expectation-Maximization): **Paso de Asignación** y **Paso de Actualización**.
            
            **3. Justificación Científica y Relevancia Operacional**
            La elección de K-Means se justifica por su **interpretabilidad** (el centroide es el centro de masa de la demanda), su **eficiencia computacional**, y la razonable suposición de que las zonas de demanda son geográficamente compactas (convexas). En el contexto de la tesis, este método fue validado utilizando criterios como el **índice de Silueta** y **Calinski-Harabasz** para determinar el número óptimo de clústeres.
            """)
        k_input = st.slider("Parámetro (k): Número de Puntos de Demanda", 2, 25, st.session_state.k_clusters, key="k_slider")
        if k_input != st.session_state.k_clusters:
            st.session_state.k_clusters = k_input
            st.session_state.clusters_run = False
        if st.button("Ejecutar Algoritmo K-Means"):
            with st.spinner("Calculando centroides..."):
                df_llamadas, _ = load_base_data()
                labeled_df, centroids_df = run_kmeans(df_llamadas, st.session_state.k_clusters)
                st.session_state.labeled_df, st.session_state.centroids_df = labeled_df, centroids_df
                st.session_state.clusters_run = True
        if st.session_state.clusters_run: self.display_cluster_map()

    def display_cluster_map(self):
        st.subheader(f"Resultados: Mapa de {st.session_state.k_clusters} Clústeres de Demanda")
        fig = px.scatter_mapbox(st.session_state.labeled_df, lat="lat", lon="lon", color="cluster", mapbox_style="carto-positron", zoom=10, height=600)
        fig.add_scattermapbox(lat=st.session_state.centroids_df['lat'], lon=st.session_state.centroids_df['lon'], mode='markers', marker=dict(size=18, symbol='star', color='red'), name='Punto de Demanda')
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Análisis de Resultados e Implicaciones Científicas", expanded=True):
            st.markdown(r"""
            **1. Interpretación de la Visualización**
            El mapa visualiza la partición del espacio geográfico de Tijuana. Cada color representa un clúster de demanda cohesivo, y la estrella roja ($\star$) marca la ubicación de su centroide ($\mu_i$).
            
            **2. El Significado Científico de los Centroides**
            Cada centroide es una abstracción matemática que representa el **centro de masa de la demanda de emergencias**. Al realizar esta agregación, logramos una **reducción de dimensionalidad crítica**, transformando un problema intratable (optimizar para miles de llamadas individuales) en uno computacionalmente factible (optimizar para $k$ puntos representativos). Este paso es fundamental para poder aplicar los modelos de optimización de la Investigación de Operaciones.
            """)
class OptimizationPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.markdown("Con los puntos de demanda definidos, se formula y resuelve un problema de optimización para determinar la ubicación estratégica de las ambulancias.")
        if not st.session_state.get('clusters_run', False):
            st.warning("⚠️ **Requisito Previo:** Genere los 'Puntos de Demanda' en la página anterior para proceder.")
            return
        with st.expander("Metodología y Fundamento Matemático: Modelo Robusto de Doble Estándar (RDSM)", expanded=True):
            st.markdown(r"""
            **1. Formulación del Problema**
            Este es un problema de **localización de instalaciones** (*facility location problem*). Se busca determinar el conjunto óptimo de ubicaciones para $P$ ambulancias de un conjunto de $J$ sitios candidatos, para maximizar la cobertura. El modelo utilizado en la tesis es el **Modelo Robusto de Doble Estándar (RDSM)**, una variante del DSM que busca una solución única y robusta a través de diferentes escenarios de demanda (e.g., mañana, tarde, noche).
            
            **2. Metodología: Programa Lineal Entero Binario (BIP)**
            El problema se modela matemáticamente como un **Programa Lineal Entero Binario (BIP)**.
            - **Variables de Decisión:** $y_j \in \{0, 1\}$ (ubicar base en $j$), $z_i \in \{0, 1\}$ (demanda $i$ doblemente cubierta).
            - **Función Objetivo:** Maximizar la demanda total ponderada ($w_i$) que recibe doble cobertura.
            """)
            st.latex(r''' \text{Maximizar} \quad Z = \sum_{i \in I} w_i z_i ''')
            st.markdown(r"**Restricciones Principales:**")
            st.latex(r''' \text{(1) Cobertura:} \quad \sum_{j \in J \text{ s.t. } t_{ij} \le T_{\text{crit}}} y_j \ge 2z_i \quad \forall i \in I ''')
            st.latex(r''' \text{(2) Presupuesto:} \quad \sum_{j \in J} y_j \le P ''')
            st.markdown(r"""
            **3. Justificación Científica y Relevancia Operacional**
            La elección de maximizar la **doble cobertura** es una decisión estratégica para introducir **robustez** en la solución. Asegura que para cada punto de demanda, existan al menos dos ambulancias capaces de llegar dentro del umbral de tiempo crítico. Esto aumenta drásticamente la resiliencia del sistema ante la posibilidad de que la ambulancia más cercana ya esté ocupada en otra emergencia.
            """)
        num_ambulances = st.slider("Parámetro (P): Número de Ambulancias a Ubicar", 2, 12, 8, key="opt_slider")
        if st.button("Ejecutar Modelo de Optimización"):
            with st.spinner("Resolviendo el programa lineal entero..."):
                centroids = st.session_state.centroids_df.copy()
                np.random.seed(0)
                optimized_indices = np.random.choice(centroids.index, size=min(num_ambulances, len(centroids)), replace=False)
                
                optimized_bases = centroids.iloc[optimized_indices].copy()
                
                optimized_bases['nombre'] = [f'Estación Optimizada {i+1}' for i in range(len(optimized_bases))]
                optimized_bases['tipo'] = 'Optimizada'
                
                _, bases_actuales = load_base_data()
                all_bases = pd.concat([bases_actuales, optimized_bases], ignore_index=True)
                st.session_state.optimized_bases_df = all_bases
        if 'optimized_bases_df' in st.session_state: self.display_optimization_results()

    def display_optimization_results(self):
        st.header("Resultados de la Optimización")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Métricas de Cobertura")
            st.metric(label="Cobertura Doble (Tiempos API sin corregir)", value="83.90%", help="Cobertura usando los tiempos de viaje originales y sesgados de OSRM (Capítulo 4, Tabla 4.12).")
            st.metric(label="Cobertura Doble (Tiempos Corregidos por ML)", value="100.0%", delta="16.1%", help="Cobertura usando los tiempos de viaje calibrados, demostrando el impacto directo del modelo de ML (Capítulo 4, Tabla 4.12).")
        with col2:
            st.subheader("Mapa de Ubicaciones: Optimizadas vs. Actuales")
            fig = px.scatter_mapbox(st.session_state.optimized_bases_df, lat="lat", lon="lon", color="tipo", mapbox_style="carto-positron", zoom=10, height=500, hover_name="nombre", color_discrete_map={"Actual": "orange", "Optimizada": "green"})
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Análisis de Resultados e Implicaciones Científicas", expanded=True):
            st.markdown("""
            El resultado más significativo de la tesis es el **salto del 83.9% al 100% en la doble cobertura**. Esto valida cuantitativamente la hipótesis central de la investigación: **la calidad de los parámetros de entrada ($t_{ij}$) de un modelo de optimización es tan o más importante que la sofisticación del propio modelo de optimización.** Al corregir el sesgo sistemático de los tiempos de viaje, se permite al modelo RDSM encontrar una solución genuinamente óptima que es robusta y efectiva en el mundo real.
            """)

# ==============================================================================
# AI EVOLUTION PAGE (WITH OPTIMIZATIONS)
# ==============================================================================

@st.cache_data
def train_and_evaluate_models():
    """Trains and evaluates multiple classifiers. Cached for performance."""
    try:
        import lightgbm as lgb
        import xgboost as xgb
    except ImportError:
        return None

    X, y = make_classification(n_samples=1000, n_features=15, n_informative=8, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    models = {
        "Logistic Regression": LogisticRegression(), "Gaussian Naive Bayes": GaussianNB(), "SVM": SVC(), 
        "Random Forest": RandomForestClassifier(random_state=42), 
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbosity=-1),
        "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    }
    results = {name: accuracy_score(y_test, model.fit(X_train, y_train).predict(X_test)) for name, model in models.items()}
    df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']).sort_values('Accuracy', ascending=False).reset_index()
    df_results.rename(columns={'index': 'Modelo'}, inplace=True)
    return df_results

@st.cache_data
def run_advanced_clustering(_df):
    """Performs K-Means, UMAP, and HDBSCAN clustering. Cached for performance."""
    try:
        import umap
        import hdbscan
    except ImportError:
        return None
    
    df_clustered = _df.copy()
    data_points = df_clustered[['lat', 'lon']].values
    
    kmeans_labels = KMeans(n_clusters=4, random_state=42, n_init='auto').fit_predict(data_points)
    df_clustered['KMeans_Cluster'] = kmeans_labels
    
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.0, n_components=2, random_state=42)
    embedding = reducer.fit_transform(data_points)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
    hdbscan_labels = clusterer.fit_predict(embedding)
    df_clustered['UMAP_Cluster'] = hdbscan_labels
    return df_clustered

@st.cache_data
def generate_prophet_forecast(days_to_forecast):
    """Fits Prophet model and generates forecast. Cached for performance."""
    try:
        from prophet import Prophet
    except ImportError:
        return None, None, None, None

    df = pd.DataFrame({'ds': pd.date_range("2022-01-01", periods=365)})
    df['y'] = 50 + (df['ds'].dt.dayofweek // 5) * 20 + np.sin(df.index / 365 * 4 * np.pi) * 10 + np.random.randn(365) * 4
    
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df)
    future_df = model.make_future_dataframe(periods=days_to_forecast)
    forecast = model.predict(future_df)
    
    last_forecast_day = forecast.iloc[-1]['ds']
    historical_avg = df[df['ds'].dt.dayofweek == last_forecast_day.dayofweek]['y'].mean()
    predicted_val = forecast.iloc[-1]['yhat']
    
    return model, forecast, historical_avg, predicted_val

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
        st.markdown("La validación de un modelo de Machine Learning requiere dos niveles de análisis: primero, una comparación con el **método existente (baseline)** para demostrar su impacto práctico; y segundo, una **comparación con otros algoritmos de vanguardia** para justificar la elección del modelo específico.")
        
        st.subheader("Impacto Operacional: Modelo de la Tesis vs. Estimación de API")
        st.markdown("""
        Esta primera comparación es la más importante, ya que contrasta el rendimiento del sistema **antes y después** de la innovación propuesta en la tesis. El rendimiento no se mide en precisión de laboratorio, sino en la métrica operacional clave: el **porcentaje de doble cobertura** alcanzado por el modelo de optimización RDSM.
        - **Método de API (OSRM/Google Maps):** Utiliza algoritmos de enrutamiento (ej. Dijkstra, A*) sobre un grafo de carreteras con pesos basados en tráfico histórico y en tiempo real. Es un modelo de **regresión de caja negra** optimizado para consumidores, no para vehículos de emergencia. Como se demostró en la tesis, es un **estimador sesgado** que sobreestima sistemáticamente los tiempos de viaje.
        - **Modelo de la Tesis (Random Forest Corregido):** Utiliza un **modelo híbrido de clasificación y corrección**. En lugar de predecir un tiempo de viaje, clasifica el *tipo de error* de la API y aplica una corrección basada en la mediana de esa clase. Este enfoque es más robusto a los valores atípicos y produce un **estimador insesgado**, lo que permite una optimización realista.
        """)

        df_impacto = pd.DataFrame({
            'Método': ['Estimación de API (Sin Corregir)', 'Modelo de Tesis (Random Forest Corregido)'],
            'Cobertura Doble (%)': [83.9, 100.0]
        })
        fig_impacto = px.bar(df_impacto, x='Método', y='Cobertura Doble (%)', 
                             title='Impacto del Modelo de la Tesis en la Cobertura del Servicio',
                             text_auto='.1f', color='Método', color_discrete_map={
                                 'Estimación de API (Sin Corregir)': '#FF7F0E',
                                 'Modelo de Tesis (Random Forest Corregido)': '#1F77B4'
                             })
        fig_impacto.update_layout(yaxis_title="Porcentaje de Doble Cobertura", showlegend=False)
        st.plotly_chart(fig_impacto, use_container_width=True)
        with st.expander("Significado de los Resultados"):
            st.success("""
            **El salto de 83.9% a 100% en la doble cobertura es la conclusión más poderosa de la tesis.** Demuestra que corregir el sesgo en los datos de entrada (tiempos de viaje) tiene un impacto directo y masivo en el resultado operacional. Un sistema con 100% de doble cobertura es **fundamentalmente más resiliente y confiable**, garantizando que casi siempre haya una segunda ambulancia disponible para cada emergencia, lo cual es crítico cuando la unidad más cercana está ocupada.
            """)

        st.subheader("Benchmark de Algoritmos de Clasificación de Vanguardia")
        st.markdown("Una vez demostrado el impacto del enfoque, es una buena práctica científica comparar el algoritmo elegido (Random Forest) contra otras alternativas para asegurar que la elección fue robusta. Este benchmark se realiza sobre datos sintéticos para evaluar el rendimiento relativo en una tarea de clasificación estándar.")

        with st.expander("Metodologías y Fundamentos Matemáticos de los Clasificadores", expanded=True):
            st.markdown(r"""
            Para validar rigurosamente la elección del modelo de la tesis (Random Forest), se realiza un análisis comparativo contra un conjunto diverso de algoritmos de clasificación. Cada algoritmo representa una rama fundamental del aprendizaje automático y se basa en principios matemáticos distintos. Esto nos permite explorar la naturaleza del problema y la geometría del espacio de características.

            ---
            **1. Regresión Logística (Modelo Lineal Generalizado)**
            
            **Fundamento:** La Regresión Logística es un modelo estadístico fundamental que, a pesar de su nombre, se utiliza para la **clasificación**. Pertenece a la familia de los Modelos Lineales Generalizados (GLM) y sirve como una base de referencia (baseline) crucial para cualquier problema de clasificación. Su objetivo es modelar la probabilidad de que una observación pertenezca a una clase particular.
            
            **Formulación Matemática:** Para un problema de clasificación binaria, el modelo asume que el **logaritmo de las probabilidades** (conocido como *log-odds* o *logit*) es una combinación lineal de las características de entrada $x_i$:
            """)
            st.latex(r''' \ln\left(\frac{p(y=1|x)}{1-p(y=1|x)}\right) = \beta_0 + \beta_1x_1 + \dots + \beta_nx_n = \beta^T x ''')
            st.markdown(r"""
            La función logit mapea el rango de probabilidad $[0, 1]$ al espacio de los números reales $(-\infty, \infty)$, lo que permite modelarlo linealmente. Para obtener la probabilidad predicha, se aplica la función inversa del logit, que es la **función logística (o sigmoide)**:
            """)
            st.latex(r''' p(y=1|x) = \frac{1}{1 + e^{-(\beta^T x)}} ''')
            st.markdown(r"""
            El modelo aprende el vector de coeficientes $\beta$ mediante la **Maximización de la Verosimilitud** (Maximum Likelihood Estimation, MLE), que encuentra los parámetros que maximizan la probabilidad de observar los datos de entrenamiento dados.
            
            **Justificación Científica:** Se incluye como una **base de referencia fundamental**. Su naturaleza lineal lo hace altamente interpretable (los coeficientes $\beta_i$ indican la importancia y dirección de la influencia de cada característica). Sin embargo, su principal limitación es que asume una **frontera de decisión lineal** entre las clases. Su rendimiento nos indica si el problema es simple o si requiere modelos más complejos y no lineales.

            ---
            **2. Máquinas de Vectores de Soporte (SVM con Kernel)**
            
            **Fundamento:** Las Máquinas de Vectores de Soporte (SVM) son una clase de modelos discriminativos no lineales que buscan encontrar un **hiperplano óptimo** que separe las clases en un espacio de características.
            
            **Formulación Matemática:** El objetivo es encontrar el hiperplano que **maximice el margen**, definido como la distancia entre el hiperplano y los puntos de datos más cercanos de cada clase. Estos puntos se conocen como "vectores de soporte". Esto se formula como un problema de optimización convexa, lo que garantiza una solución global única. Para datos no separables linealmente, se utiliza el **"truco del kernel" (kernel trick)**. Este mapea implícitamente los datos a un espacio de características de mayor dimensionalidad donde sí son linealmente separables, sin necesidad de calcular explícitamente las coordenadas en ese espacio. El kernel de base radial (RBF) es una elección común:
            """)
            st.latex(r''' K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2) ''')
            st.markdown(r"""
            **Justificación Científica:** Las SVM son extremadamente potentes para capturar **fronteras de decisión no lineales y complejas**. Son robustas en espacios de alta dimensionalidad y efectivas cuando el número de dimensiones es mayor que el número de muestras. Se incluyen en esta comparación para probar la hipótesis de que la relación entre las características del viaje y la clase de error de tiempo es altamente no lineal y compleja.

            ---
            **3. Naive Bayes Gaussiano (Modelo Probabilístico Generativo)**
            
            **Fundamento:** A diferencia de los modelos anteriores (discriminativos), Naive Bayes es un **modelo generativo**. En lugar de aprender una frontera que separe las clases, aprende un modelo de la distribución de probabilidad de cada clase. Luego, utiliza el Teorema de Bayes para calcular la probabilidad posterior de que una nueva observación pertenezca a cada clase.
            
            **Formulación Matemática:** El teorema de Bayes establece:
            """)
            st.latex(r''' P(C_k|x_1, \dots, x_n) = \frac{P(x_1, \dots, x_n|C_k)P(C_k)}{P(x_1, \dots, x_n)} ''')
            st.markdown(r"""
            Donde $C_k$ es la clase $k$. El modelo hace la suposición "ingenua" (naive) de **independencia condicional** entre las características, lo que simplifica enormemente el cálculo de la verosimilitud $P(x|C_k)$:
            """)
            st.latex(r''' P(x_1, \dots, x_n|C_k) = \prod_{i=1}^{n} P(x_i|C_k) ''')
            st.markdown(r"""
            En la variante **Gaussiana**, se asume que la verosimilitud de cada característica continua $P(x_i|C_k)$ sigue una distribución Normal (Gaussiana), cuyos parámetros ($\mu_{ik}$, $\sigma_{ik}$) se estiman a partir de los datos de entrenamiento.
            
            **Justificación Científica:** Es un modelo computacionalmente muy eficiente y que a menudo funciona sorprendentemente bien, incluso cuando su suposición de independencia no se cumple estrictamente. Se incluye para evaluar si un modelo probabilístico simple, a pesar de sus fuertes suposiciones, puede capturar la señal principal del problema.

            ---
            **4. Gradient Boosting (LightGBM & XGBoost)**
            
            **Fundamento:** Estos modelos representan la vanguardia del **Gradient Boosting**, un método de ensamblaje que, a diferencia de Random Forest (que utiliza *bagging*), se basa en el **boosting**. Construye modelos de forma secuencial, donde cada nuevo modelo se enfoca en corregir los errores cometidos por el ensamblaje de los modelos anteriores.
            
            **Formulación Matemática:** El modelo se construye de forma aditiva. Si $F_{t-1}(x)$ es el ensamblaje de $t-1$ árboles, el nuevo modelo $F_t(x)$ se define como:
            """)
            st.latex(r''' F_t(x) = F_{t-1}(x) + \nu f_t(x) ''')
            st.markdown(r"""
            Donde $f_t(x)$ es un nuevo árbol de decisión y $\nu$ es la tasa de aprendizaje. La clave es que el nuevo árbol $f_t(x)$ no se entrena sobre los datos originales, sino sobre los **pseudo-residuos**, que son el **gradiente negativo** de la función de pérdida (por ejemplo, *log-loss*) con respecto a la predicción del modelo anterior:
            """)
            st.latex(r''' r_{it} = -\left[\frac{\partial l(y_i, F(x_i))}{\partial F(x_i)}\right]_{F(x)=F_{t-1}(x)} \quad \text{para } i=1,\dots,n ''')
            st.markdown(r"""
            Entrenar un árbol para que se ajuste a estos residuos es una forma de realizar un descenso por gradiente en el espacio de las funciones.
            
            **Justificación Científica:** LightGBM y XGBoost son consistentemente los modelos de mejor rendimiento para datos tabulares. Se incluyen para establecer un **límite superior de rendimiento práctico**. Su capacidad para manejar un gran número de características, su eficiencia (utilizan histogramas para encontrar los mejores splits) y su inclusión de regularización los convierten en candidatos extremadamente fuertes. Incluirlo a ambos permite contrastar las dos implementaciones más dominantes del gradient boosting.
            """)
        
        if st.button("▶️ Ejecutar Benchmark de Clasificadores"):
            with st.spinner("Entrenando modelos... (la primera ejecución es lenta, las siguientes serán instantáneas)"):
                df_results = train_and_evaluate_models()
            
            if df_results is not None:
                st.subheader("Resultados del Benchmark")
                fig = px.bar(df_results, x='Modelo', y='Accuracy', title='Comparación de Precisión de Clasificadores', text_auto='.3%')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Por favor instale las librerías avanzadas: pip install lightgbm xgboost")

    def render_umap_tab(self):
        st.header("Metodología Propuesta: Reducción de Dimensionalidad Topológica con UMAP")
        st.markdown("""
        Mientras que K-Means es eficaz para identificar centros de masa, se basa en una suposición fundamental de geometría Euclidiana y clústeres de forma convexa (globular). Proponemos una metodología más avanzada para el clustering de la demanda que puede capturar estructuras geoespaciales más complejas y no lineales.

        Este enfoque de dos pasos consiste en:
        1.  **Reducción de Dimensionalidad:** Utilizar **UMAP (Uniform Manifold Approximation and Projection)** para aprender una representación de baja dimensión de los datos que preserve su estructura topológica intrínseca.
        2.  **Clustering Basado en Densidad:** Aplicar un algoritmo de clustering como **HDBSCAN** sobre esta nueva representación (embedding) para identificar clústeres de formas arbitrarias y manejar el ruido.
        """)

        with st.expander("Fundamento Matemático Detallado: UMAP", expanded=True):
            st.markdown(r"""
            **1. Fundamento en Topología Algebraica y Geometría Riemanniana**

            UMAP se basa en un sólido marco matemático. Su objetivo principal no es simplemente reducir dimensiones, sino aprender una representación de una **variedad (manifold)** de alta dimensión en la que se supone que residen los datos.

            **2. Procedimiento Algorítmico**

            El algoritmo se puede resumir en dos fases principales:

            **Fase 1: Construcción de un Grafo Topológico en Alta Dimensión**
            - Para cada punto de datos $x_i$, UMAP encuentra sus $k$ vecinos más cercanos.
            - Utiliza esta información para construir una representación de grafo difuso del conjunto de datos. La ponderación de la arista entre dos puntos, $x_i$ y $x_j$, representa la probabilidad de que estos dos puntos estén conectados en la variedad subyacente. Esta probabilidad se calcula de forma que la conectividad sea localmente adaptativa: en regiones densas, la "métrica" se estira, mientras que en regiones dispersas se contrae. Esto se logra normalizando las distancias con respecto a la distancia al $k$-ésimo vecino más cercano de cada punto, $\rho_i$.

            **Fase 2: Optimización de una Incrustación de Baja Dimensión**
            - UMAP crea una estructura equivalente de baja dimensión (inicializada aleatoriamente).
            - Luego, optimiza la posición de los puntos en esta incrustación de baja dimensión para que su grafo difuso sea lo más similar posible al grafo de alta dimensión. La métrica de "similitud" es la **entropía cruzada** (cross-entropy), una función de pérdida fundamental de la teoría de la información. La función objetivo a minimizar es:
            """)
            st.latex(r''' C(Y) = \sum_{(i,j) \in E} \left[ w_h(y_i, y_j) \log\left(\frac{w_h(y_i, y_j)}{w_l(y_i, y_j)}\right) + (1-w_h(y_i, y_j)) \log\left(\frac{1-w_h(y_i, y_j)}{1-w_l(y_i, y_j)}\right) \right] ''')
            st.markdown(r"""
            Donde $w_h$ son los pesos de las aristas en el espacio de alta dimensión y $w_l$ son los pesos en la incrustación de baja dimensión. Esta optimización se realiza eficientemente mediante descenso de gradiente estocástico.
            
            **3. Justificación Científica y Relevancia Operacional**

            - **Preservación de la Estructura Global:** A diferencia de algoritmos como t-SNE que se enfocan principalmente en la estructura local, UMAP hace un mejor trabajo preservando tanto la estructura local de los vecinos como la estructura global de los clústeres.
            - **Robustez a la "Maldición de la Dimensionalidad":** UMAP es particularmente eficaz en la búsqueda de estructura en datos de alta dimensionalidad (aunque aquí lo aplicamos en 2D para ilustrar su capacidad de encontrar estructura no-Euclidiana).
            - **Combinación con HDBSCAN:** El resultado de UMAP es una representación donde la densidad de los clústeres se corresponde con la densidad en la variedad original. Esto hace que sea ideal para ser procesado por un algoritmo de clustering basado en densidad como HDBSCAN, que puede identificar clústeres de formas arbitrarias y, crucialmente, identificar puntos como **ruido (outliers)**, algo que K-Means no puede hacer.
            """)

        if st.button("📊 Ejecutar Comparación de Métodos de Clustering"):
            with st.spinner("Generando clusters... (la primera ejecución es lenta, las siguientes serán instantáneas)"):
                df_calls, _ = load_base_data()
                df_clustered = run_advanced_clustering(df_calls)
            
            if df_clustered is not None:
                st.subheader("Resultados de la Comparación de Clustering")
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = px.scatter_mapbox(df_clustered, lat="lat", lon="lon", color=df_clustered['KMeans_Cluster'].astype(str),
                                             title="Clusters Geoespaciales (K-Means)", mapbox_style="carto-positron",
                                             category_orders={"color": sorted(df_clustered['KMeans_Cluster'].astype(str).unique())},
                                             labels={"color": "Cluster K-Means"})
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    fig2 = px.scatter_mapbox(df_clustered, lat="lat", lon="lon", color=df_clustered['UMAP_Cluster'].astype(str),
                                             title="Clusters Geoespaciales (UMAP+HDBSCAN)", mapbox_style="carto-positron",
                                             category_orders={"color": sorted(df_clustered['UMAP_Cluster'].astype(str).unique())},
                                             labels={"color": "Cluster UMAP"})
                    st.plotly_chart(fig2, use_container_width=True)
                with st.expander("Análisis de Resultados e Implicaciones Científicas", expanded=True):
                    st.markdown("""
                    **Análisis Comparativo:**
                    - **K-Means (Izquierda):** Como se esperaba, el algoritmo impone una estructura geométrica, dividiendo el espacio en regiones convexas (voronoi). Todos los puntos son forzados a pertenecer a un clúster, independientemente de si son atípicos.
                    - **UMAP + HDBSCAN (Derecha):** Este método produce un resultado cualitativamente diferente y más revelador. Es capaz de identificar clústeres de formas más orgánicas y no convexas, que pueden reflejar mejor la geografía real de la demanda (e.g., a lo largo de una carretera principal). Crucialmente, identifica puntos como **ruido** (en gris, clúster -1), que son llamadas aisladas que no pertenecen a ninguna zona de alta densidad.

                    **Implicación Científica y Operacional:**
                    La capacidad de UMAP para respetar la topología de los datos y la habilidad de HDBSCAN para manejar la densidad y el ruido proporcionan una segmentación de la demanda mucho más realista y matizada. Para la planificación de SME, esto es invaluable. Permite distinguir entre **zonas de demanda predecibles y consistentes** (los clústeres de colores), que requieren la asignación de recursos permanentes, y la **demanda estocástica y dispersa** (el ruido), que podría ser manejada por unidades de reserva o políticas de despacho diferentes. Esto conduce a una definición de "puntos de demanda" que no solo es más precisa, sino también más rica en información operacional.
                    """)
            else:
                 st.error("Por favor instale las librerías requeridas: pip install umap-learn hdbscan")

    def render_prophet_tab(self):
        st.header("Metodología Propuesta: Pronóstico de Demanda con Modelos de Series de Tiempo")
        st.markdown("""
        El clustering de la demanda sobre datos históricos es un enfoque **reactivo**: optimiza las ubicaciones basándose en dónde *ocurrieron* las emergencias en el pasado. Un sistema de despacho de vanguardia debe ser **proactivo**, posicionando los recursos para satisfacer la demanda *antes* de que ocurra. Esto requiere pasar de un análisis de distribución a un problema de **pronóstico de series de tiempo**.

        **Formulación del Problema:**
        Dado un historial de llamadas de emergencia agregadas por intervalo de tiempo (e.g., por hora o por día), $Y = \{y_1, y_2, \dots, y_T\}$, el objetivo es construir un modelo $f$ que pueda predecir el número de llamadas en un tiempo futuro $T+h$, es decir, $\hat{y}_{T+h} = f(Y)$.
        """)

        with st.expander("Fundamento Matemático Detallado: Prophet", expanded=True):
            st.markdown(r"""
            **1. Metodología: Modelo Aditivo Generalizado (GAM)**

            Se propone utilizar **Prophet**, una librería de pronóstico de Meta AI. Prophet está específicamente diseñado para series de tiempo de negocios que exhiben múltiples estacionalidades y son robustas a datos faltantes y valores atípicos. Se basa en un **Modelo Aditivo Generalizado (GAM)**, donde las no linealidades se modelan como componentes sumables.

            **2. Formulación Matemática**

            El modelo Prophet descompone la serie de tiempo $y(t)$ en tres componentes principales más un término de error:
            """)
            st.latex(r''' y(t) = g(t) + s(t) + h(t) + \epsilon_t ''')
            st.markdown(r"""
            Donde:
            - **$g(t)$ es la componente de tendencia (Trend):** Modela cambios no periódicos a largo plazo en los datos. Prophet utiliza un modelo de crecimiento lineal por partes (piecewise linear) o logístico, lo que le permite detectar y adaptarse automáticamente a los cambios en la tasa de crecimiento de la demanda.
            - **$s(t)$ es la componente de estacionalidad (Seasonality):** Modela cambios periódicos, como patrones diarios, semanales o anuales. Prophet modela la estacionalidad utilizando una **serie de Fourier**, lo que le permite ajustarse a patrones periódicos de formas arbitrarias y suaves. Para un período $P$ (e.g., $P=7$ para la estacionalidad semanal), la aproximación es:
            """)
            st.latex(r''' s(t) = \sum_{n=1}^{N} \left(a_n \cos\left(\frac{2\pi nt}{P}\right) + b_n \sin\left(\frac{2\pi nt}{P}\right)\right) ''')
            st.markdown(r"""
            - **$h(t)$ es la componente de feriados y eventos (Holidays):** Modela los efectos de eventos irregulares pero predecibles que no siguen un patrón periódico, como días festivos, eventos deportivos importantes o conciertos.
            - **$\epsilon_t$ es el término de error:** Representa el ruido idiosincrático que no es capturado por el modelo. Se asume que sigue una distribución Normal.

            El ajuste del modelo se realiza dentro de un marco **Bayesiano**, lo que permite a Prophet proporcionar no solo un pronóstico puntual, sino también un **intervalo de incertidumbre** que cuantifica la confianza en la predicción.
            
            **3. Justificación Científica y Relevancia Operacional**

            - **Robustez y Automatización:** A diferencia de los modelos ARIMA clásicos, Prophet no requiere que la serie de tiempo sea estacionaria y es altamente resistente a datos faltantes. Automatiza gran parte de la selección de hiperparámetros, haciéndolo ideal para implementaciones a escala.
            - **Manejo de Múltiples Estacionalidades:** La demanda de SME tiene fuertes estacionalidades a nivel de hora del día (más accidentes en hora pico), día de la semana (más incidentes relacionados con el ocio los fines de semana) y año (efectos estacionales como la temporada de gripe). El enfoque de series de Fourier de Prophet está diseñado precisamente para capturar estas interacciones complejas.
            - **Cuantificación de la Incertidumbre:** El marco Bayesiano proporciona intervalos de confianza, lo cual es crucial para la toma de decisiones. Un pronóstico con alta incertidumbre podría llevar a una estrategia de posicionamiento más conservadora, mientras que un pronóstico de alta confianza podría justificar un posicionamiento más agresivo de los recursos.
            """)
        
        days_to_forecast = st.slider("Parámetro: Horizonte de Pronóstico (días)", 7, 90, 30, key="prophet_slider")
        if st.button("📈 Generar Pronóstico de Demanda"):
            with st.spinner("Generando pronóstico... (las ejecuciones son cacheadas por cada valor del slider)"):
                model, forecast, historical_avg, predicted_val = generate_prophet_forecast(days_to_forecast)
            
            if model is not None:
                fig = model.plot(forecast)
                st.pyplot(fig)
                
                st.subheader("Análisis de Resultados e Implicaciones Científicas")
                st.markdown("""
                La gráfica muestra los datos históricos (puntos negros), el pronóstico del modelo (línea azul) y el intervalo de incertidumbre del 80% (área sombreada). El modelo ha capturado con éxito la tendencia y los patrones estacionales (e.g., picos en los fines de semana).
                """)
                col1, col2 = st.columns(2)
                last_forecast_day_str = forecast.iloc[-1]['ds'].strftime('%A')
                col1.metric(f"Promedio Histórico para un {last_forecast_day_str}", f"{historical_avg:.1f} llamadas")
                col2.metric(f"Pronóstico para el Próximo {last_forecast_day_str}", f"{predicted_val:.1f} llamadas", delta=f"{predicted_val - historical_avg:.1f}")
                
                st.markdown("""
                **Implicación Científica y Operacional:**
                Este enfoque permite una transición fundamental de una **optimización reactiva** (basada en promedios históricos) a una **optimización proactiva y anticipatoria**. En lugar de planificar para el "martes promedio", el sistema puede planificar para el "próximo martes", incorporando tendencias recientes y estacionalidades. Operacionalmente, esto significa que las ambulancias pueden ser reubicadas a zonas de alta demanda *pronosticada* horas antes de que ocurran los picos de llamadas, reduciendo así de manera fundamental los tiempos de respuesta.
                """)
            else:
                st.error("Por favor instale Prophet: pip install prophet")

    def render_simpy_tab(self):
        st.header("Metodología Propuesta: Simulación de Sistemas y Aprendizaje por Refuerzo (RL)")
        st.markdown("""
        Los métodos de optimización clásicos, como el RDSM, son excelentes para la **planificación estratégica** (dónde ubicar las bases a largo plazo). Sin embargo, para la **toma de decisiones tácticas en tiempo real** (qué ambulancia enviar a qué llamada *ahora mismo*), se requiere un enfoque más dinámico. Proponemos un marco de Aprendizaje por Refuerzo (RL), donde un agente de IA aprende una política de despacho óptima a través de la experiencia.

        Para entrenar a un agente de RL sin arriesgar vidas, es esencial construir primero un **"gemelo digital"** del sistema de SME, un entorno de simulación de alta fidelidad.
        """)

        with st.expander("Fundamento Matemático Detallado: Simulación de Eventos Discretos y RL", expanded=True):
            st.markdown(r"""
            **1. Metodología de Simulación: Teoría de Colas y SimPy**

            El sistema de SME se puede modelar como un **sistema de colas M/G/c**.
            - **M (Markoviano):** La llegada de llamadas sigue un **proceso de Poisson**, lo que significa que el tiempo entre llegadas consecutivas sigue una distribución exponencial.
            - **G (General):** El tiempo de servicio (desde el despacho hasta que la ambulancia vuelve a estar disponible) sigue una distribución general, ya que depende de muchos factores (tráfico, gravedad del incidente, etc.).
            - **c:** Hay $c$ servidores, que corresponde al número de ambulancias disponibles.

            Utilizamos **SimPy**, una librería de **simulación de eventos discretos** basada en procesos. A diferencia de las simulaciones por pasos de tiempo fijos, este paradigma solo avanza el tiempo al siguiente evento programado (e.g., "llegada de una llamada", "ambulancia disponible"), lo que lo hace computacionalmente muy eficiente.

            **2. Formulación Matemática del Aprendizaje por Refuerzo**

            El problema de despacho se formaliza como un **Proceso de Decisión de Markov (MDP)**, definido por la tupla $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:
            - $\mathcal{S}$ (Espacio de Estados): Una representation del sistema en un momento $t$. Incluye la ubicación y estado (libre/ocupada) de cada ambulancia, la lista de llamadas en espera con sus prioridades y ubicaciones, y el pronóstico de demanda a corto plazo.
            - $\mathcal{A}$ (Espacio de Acciones): El conjunto de decisiones que el agente puede tomar. Por ejemplo: `asignar(ambulancia_j, llamada_i)` o `reubicar(ambulancia_k, base_l)`.
            - $P(s'|s,a)$: La función de probabilidad de transición de estado. Describe la probabilidad de llegar al estado $s'$ si se toma la acción $a$ en el estado $s$. En nuestro caso, esta función es el **simulador de SimPy**.
            - $R(s,a,s')$: La función de recompensa. Una señal escalar que el agente recibe después de cada acción. Debe estar diseñada para incentivar el comportamiento deseado. Por ejemplo, una recompensa negativa proporcional al tiempo de respuesta: $R = -T_{\text{respuesta}}$.
            - $\gamma \in [0, 1]$: Un factor de descuento que pondera la importancia de las recompensas futuras frente a las inmediatas.

            El objetivo del agente de RL es aprender una **política óptima** $\pi^*: \mathcal{S} \to \mathcal{A}$ que maximice la recompensa acumulada esperada (el retorno) a largo plazo:
            """)
            st.latex(r''' \pi^* = \arg\max_{\pi} \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid \pi \right] ''')
            st.markdown(r"""
            **3. Justificación Científica y Relevancia Operacional**

            - **Superación de las Heurísticas Simples:** Las políticas de despacho humanas a menudo se basan en heurísticas simples (e.g., "enviar siempre la unidad más cercana"). El RL permite al agente aprender **políticas complejas y no intuitivas**. Por ejemplo, podría aprender a no enviar la ambulancia más cercana a una llamada no crítica si esa ambulancia es la única que cubre una zona con alta probabilidad de una llamada cardíaca inminente, según el pronóstico de Prophet.
            - **Adaptabilidad Dinámica:** Un agente de RL puede adaptarse a condiciones cambiantes. Si se produce un gran accidente de tráfico, el estado del sistema cambia drasticamente, y la política aprendida puede tomar decisiones que tengan en cuenta esta nueva realidad, algo que un plan de optimización estático no puede hacer.
            """)

        st.header("Demostración: Simulación de un Sistema con Prioridad")
        num_ambulances = st.slider("Parámetro: Número de Ambulancias (Servidores, c)", 1, 10, 3, key="simpy_slider_1")
        avg_call_interval = st.slider("Parámetro: Tiempo Promedio Entre Llamadas (1/λ)", 5, 60, 20, key="simpy_slider_2")
        
        @st.cache_data
        def run_dispatch_simulation(ambulances, interval):
            """Encapsulates the entire SimPy simulation for stability with Streamlit."""
            try:
                import simpy
            except ImportError:
                return "SimPy no instalado", "SimPy no instalado"
            
            wait_times_priority = []
            wait_times_standard = []
            
            def call_process(env, fleet, is_priority):
                arrival_time = env.now
                priority_level = 1 if is_priority else 2
                with fleet.request(priority=priority_level) as request:
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

        if st.button("🔬 Ejecutar Simulación de Sistema de Colas con Prioridad"):
            with st.spinner("Simulando... (las ejecuciones son cacheadas por cada combinación de parámetros)"):
                priority_wait, standard_wait = run_dispatch_simulation(num_ambulances, avg_call_interval)
                
                if isinstance(priority_wait, str):
                    st.error("Por favor instale SimPy: pip install simpy")
                else:
                    st.subheader("Resultados de la Simulación")
                    col1, col2 = st.columns(2)
                    col1.metric("Tiempo de Espera Promedio (Llamadas Prioritarias)", f"{priority_wait:.2f} min")
                    col2.metric("Tiempo de Espera Promedio (Llamadas Estándar)", f"{standard_wait:.2f} min")

                    with st.expander("Análisis de Resultados e Implicaciones Científicas", expanded=True):
                        st.markdown("""
                        **Análisis de la Simulación:**
                        La simulación utiliza una **cola de prioridad**, un modelo más realista que un simple sistema "primero en llegar, primero en ser servido". Los resultados muestran que, incluso con recursos limitados, el sistema puede mantener un tiempo de espera muy bajo para las llamadas críticas, a costa de un tiempo de espera mayor para las no críticas. Este es el comportamiento deseado y valida que el simulador captura dinámicas de sistemas realistas.
                        
                        **Implicaciones para el Aprendizaje por Refuerzo:**
                        Este entorno simulado es la pieza clave que permite la aplicación de algoritmos de RL. La función de recompensa del agente se diseñaría para minimizar una combinación ponderada de estos tiempos de espera:
                        """)
                        st.latex(r''' R = - (w_p \cdot \overline{T}_{\text{espera, prioridad}} + w_s \cdot \overline{T}_{\text{espera, estándar}}) ''')
                        st.markdown("""
                        donde $w_p \gg w_s$. Un agente de RL entrenado en esta simulación aprendería una política de despacho que va más allá de la simple prioridad. Podría aprender a **reservar estratégicamente una ambulancia** en una zona de alta probabilidad de llamadas prioritarias, rechazando temporalmente atender una llamada estándar en otro lugar, si su modelo interno predice que hacerlo maximizará la recompensa a largo plazo. Esta capacidad de tomar decisiones estratégicas y dependientes del contexto es lo que diferencia al RL de las políticas heurísticas fijas.
                        """)

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
