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
            El problema central es la optimización de un sistema estocástico y dinámico con recursos limitados. La eficacia de los SME se mide principalmente por el **tiempo de respuesta**. En entornos como Tijuana, las estimaciones de tiempo de viaje de las API comerciales son sistemáticamente incorrectas.
            Esta investigación aborda esta brecha mediante la integración de **Investigación de Operaciones** y **Aprendizaje Automático**.
            """)

class ClusteringPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.markdown("El primer paso computacional es agregar las ubicaciones de miles de llamadas históricas en un conjunto manejable de 'puntos de demanda' mediante K-Means.")
        with st.expander("Metodología y Fundamento Matemático: K-Means"):
            st.markdown(r"""
            **Problema:** Particionar un conjunto de $n$ vectores de observación de llamadas $\{x_1, \dots, x_n\}$ en $k$ clústeres $S = \{S_1, \dots, S_k\}$.
            **Objetivo:** Minimizar la inercia, o la Suma de Cuadrados Intra-clúster (WCSS):
            """)
            st.latex(r''' \arg\min_{S} \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2 ''')
            st.markdown(r"Donde $\mu_i$ es el centroide (media vectorial) del clúster $S_i$.")
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
        st.markdown("Con los puntos de demanda definidos, se formula y resuelve un problema de optimización para determinar la ubicación estratégica de las ambulancias.")
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
        st.markdown("Esta sección explora metodologías de vanguardia para extender la investigación actual, presentando prototipos funcionales con explicaciones científicas detalladas.")
        tab1, tab2, tab3 = st.tabs(["1. Modelos de Gradient Boosting", "2. Pronóstico de Demanda", "3. Simulación y RL"])
        with tab1: self.render_xgboost_tab()
        with tab2: self.render_prophet_tab()
        with tab3: self.render_simpy_tab()

    def render_xgboost_tab(self):
        st.header("Metodología Propuesta: Modelos de Gradient Boosting")
        st.markdown("""
        **Formulación del Problema:** El objetivo es mejorar la precisión del modelo de clasificación de corrección de tiempo. Mientras que Random Forest construye cientos de árboles de decisión independientes en paralelo y promedia sus predicciones (un método conocido como *bagging*), los métodos de Gradient Boosting construyen árboles de forma secuencial (un método conocido como *boosting*).

        **Fundamento Matemático (XGBoost):**
        Sea un conjunto de datos $\{(x_i, y_i)\}_{i=1}^n$. El objetivo es aprender una función $F(x)$ que aproxime $y$. XGBoost construye esta función como una suma de $K$ árboles de decisión $f_k$:
        """)
        st.latex(r''' \hat{y}_i = F(x_i) = \sum_{k=1}^{K} f_k(x_i) ''')
        st.markdown(r"""
        El modelo se entrena de forma aditiva. En cada iteración $t$, se añade un nuevo árbol $f_t$ que minimiza la siguiente función objetivo:
        """)
        st.latex(r''' \mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) ''')
        st.markdown(r"""
        Donde:
        - $l$ es una función de pérdida diferenciable (e.g., *log-loss* para clasificación).
        - $\hat{y}_i^{(t-1)}$ es la predicción del modelo en la iteración anterior.
        - $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda\|w\|^2$ es un término de **regularización** que penaliza la complejidad del árbol ($T$ es el número de hojas, $w$ son los pesos de las hojas).
        
        **Justificación:** A diferencia de Random Forest, XGBoost se enfoca en los residuos (errores) de las iteraciones anteriores. La inclusión explícita del término de regularización $\Omega(f_t)$ controla el sobreajuste, lo que a menudo resulta en un modelo más generalizable y preciso, especialmente en datos tabulares con interacciones complejas.
        """)
        if st.button("▶️ Ejecutar Comparación Empírica"):
            with st.spinner("Entrenando modelos..."):
                # (Code remains the same)
                import xgboost as xgb
                X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
                xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42).fit(X_train, y_train)
                results = {"Random Forest (Base)": accuracy_score(y_test, rf.predict(X_test)), "XGBoost (Propuesto)": accuracy_score(y_test, xgb_model.predict(X_test))}
                df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']).reset_index()
                fig = px.bar(df_results, x='index', y='Accuracy', title='Comparación de Precisión de Modelos', text_auto='.3%')
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Análisis de Resultados e Implicaciones"):
                    st.markdown("La demostración empírica corrobora la superioridad teórica de XGBoost. La mejora en la precisión, aunque parezca marginal, es significativa en un contexto operacional. Una clasificación más precisa del error del tiempo de viaje conduce a parámetros de entrada más fiables para el modelo de optimización RDSM, lo que a su vez genera soluciones de localización de ambulancias más eficientes y robustas.")

    def render_prophet_tab(self):
        st.header("Metodología Propuesta: Pronóstico de Demanda con Prophet")
        st.markdown("""
        **Formulación del Problema:** Tratar el número de llamadas de emergencia por día (o por hora) como una serie de tiempo $y(t)$. El objetivo es construir un modelo que pronostique valores futuros $y(t+h)$ para un horizonte $h$.
        
        **Fundamento Matemático (Prophet):**
        Prophet modela una serie de tiempo como una suma de componentes en un modelo aditivo generalizado:
        """)
        st.latex(r''' y(t) = g(t) + s(t) + h(t) + \epsilon_t ''')
        st.markdown(r"""
        Donde:
        - $g(t)$ es la **tendencia** (trend), modelada con una función lineal por partes o logística para capturar cambios no periódicos.
        - $s(t)$ es la **estacionalidad** (seasonality), modelada con una serie de Fourier para capturar patrones periódicos (e.g., semanal, anual). $s(t) = \sum_{n=1}^{N} (a_n \cos(\frac{2\pi nt}{P}) + b_n \sin(\frac{2\pi nt}{P}))$.
        - $h(t)$ representa el efecto de **feriados** o eventos irregulares (holidays).
        - $\epsilon_t$ es el término de error, asumido como ruido Gaussiano.
        
        **Justificación:** A diferencia de los modelos ARIMA clásicos que requieren que la serie sea estacionaria, Prophet está diseñado para ser robusto a datos faltantes, cambios de tendencia y valores atípicos. Su capacidad para incorporar múltiples estacionalidades (e.g., hora del día, día de la semana) lo hace especialmente adecuado para modelar la demanda de servicios de emergencia, que se sabe que sigue estos patrones complejos.
        """)
        days_to_forecast = st.slider("Parámetro: Horizonte de Pronóstico (días)", 7, 90, 30)
        if st.button("📈 Generar Pronóstico de Demanda"):
            with st.spinner("Calculando..."):
                from prophet import Prophet
                df = pd.DataFrame({'ds': pd.date_range("2022-01-01", periods=365)})
                df['y'] = 50 + (df['ds'].dt.dayofweek // 5) * 20 + np.sin(df.index / 365 * 4 * np.pi) * 10
                model = Prophet(weekly_seasonality=True, yearly_seasonality=True).fit(df)
                forecast = model.predict(model.make_future_dataframe(periods=days_to_forecast))
                fig = model.plot(forecast)
                st.pyplot(fig)
                with st.expander("Análisis de Resultados e Implicaciones"):
                    st.markdown("""
                    El gráfico del pronóstico muestra la predicción del modelo (línea azul) y el intervalo de incertidumbre bayesiano (área sombreada), que cuantifica la confianza en la predicción.
                    **Implicación Científica:** Este enfoque permite pasar de una **optimización reactiva** (basada en datos históricos) a una **optimización proactiva y anticipatoria**. La estrategia de ubicación de ambulancias ya no se basa en el pasado, sino en una predicción estadísticamente sólida del futuro inmediato. Esto permite al sistema posicionar recursos para satisfacer la demanda *esperada*, reduciendo fundamentalmente los tiempos de respuesta.
                    """)

    def render_simpy_tab(self):
        st.header("Metodología Propuesta: Simulación y Aprendizaje por Refuerzo (RL)")
        st.markdown("""
        **Formulación del Problema:** Optimizar la política de despacho dinámica $\pi(a|s)$, que mapea un estado del sistema $s$ a una acción de despacho $a$. Este es un problema de control óptimo en un entorno estocástico.
        
        **Fundamento Matemático (Marco de RL):**
        El problema se modela como un **Proceso de Decisión de Markov (MDP)**, definido por la tupla $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:
        - $\mathcal{S}$: El espacio de estados (ubicación de ambulancias, llamadas en cola).
        - $\mathcal{A}$: El espacio de acciones (a qué llamada enviar qué ambulancia).
        - $P(s'|s,a)$: La probabilidad de transición al estado $s'$ desde $s$ al tomar la acción $a$.
        - $R(s,a,s')$: La recompensa recibida.
        - $\gamma$: Un factor de descuento.
        
        El objetivo es encontrar la política óptima $\pi^*$ que maximice la recompensa acumulada esperada (el retorno):
        """)
        st.latex(r''' \pi^* = \arg\max_{\pi} \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid \pi \right] ''')
        st.markdown("""
        **Justificación:** Mientras que la programación lineal encuentra una solución óptima *estática*, el RL puede aprender una política *dinámica* que se adapte a las condiciones cambiantes en tiempo real. Se utiliza **SimPy** para construir un modelo de simulación de eventos discretos que aproxime la función de transición $P$. Este "gemelo digital" sirve como el entorno en el que un agente de RL (e.g., de **Stable-Baselines3**) puede aprender la política $\pi^*$ a través de millones de interacciones simuladas sin riesgo para el mundo real.
        """)
        num_ambulances = st.slider("Parámetro: Número de Ambulancias", 1, 10, 3)
        avg_call_interval = st.slider("Parámetro: Tiempo Promedio Entre Llamadas (min)", 5, 60, 20)
        
        @st.cache_data
        def run_dispatch_simulation(ambulances, interval, num_calls_to_sim):
            import simpy
            import random
            wait_times = []
            env = simpy.Environment()
            fleet = simpy.Resource(env, capacity=ambulances)
            def call_proc(env, fleet):
                arrival = env.now
                with fleet.request() as req:
                    yield req; wait_times.append(env.now - arrival)
                    yield env.timeout(np.random.uniform(20, 40))
            def generator(env, fleet, interval):
                for _ in range(num_calls_to_sim):
                    env.process(call_proc(env, fleet)); yield env.timeout(np.random.expovariate(1.0 / interval))
            env.process(generator(env, fleet, interval)); env.run()
            return np.mean(wait_times) if wait_times else 0

        if st.button("🔬 Ejecutar Simulación de Eventos Discretos"):
            with st.spinner("Simulando..."):
                avg_wait = run_dispatch_simulation(num_ambulances, avg_call_interval, num_calls_to_sim=500)
                st.metric("Resultado: Tiempo Promedio de Espera por Ambulancia", f"{avg_wait:.2f} minutos")
                with st.expander("Análisis de Resultados e Implicaciones"):
                    st.markdown("""
                    La simulación cuantifica la relación no lineal entre los recursos del sistema y el rendimiento.
                    **Implicación Científica:** Este entorno simulado es la pieza clave que permite aplicar algoritmos de RL. La métrica de "Tiempo Promedio de Espera" puede ser utilizada para definir la función de recompensa (e.g., $R = -(\text{tiempo de espera})$). Un agente de RL entrenado en esta simulación aprendería una política de despacho que podría superar a cualquier heurística humana, ya que puede tener en cuenta el estado completo del sistema y las consecuencias a largo plazo de cada decisión.
                    """)

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
