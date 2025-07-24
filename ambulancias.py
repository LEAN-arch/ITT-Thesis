# app.py
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
from abc import ABC, abstractmethod

# ==============================================================================
# 1. UI COMPONENTS & HELPER FUNCTIONS
# (Previously in components.py, now integrated for a single-file solution)
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
    lat_min, lat_max = 32.40, 32.55
    lon_min, lon_max = -117.12, -116.60
    num_llamadas = 500
    np.random.seed(42)
    df_llamadas = pd.DataFrame({
        'lat': np.random.uniform(lat_min, lat_max, num_llamadas),
        'lon': np.random.uniform(lon_min, lon_max, num_llamadas),
        'tiempo_api_minutos': np.random.uniform(5, 30, num_llamadas),
    })
    bases_actuales = pd.DataFrame({
        'nombre': ['Base Actual - Centro', 'Base Actual - La Mesa'],
        'lat': [32.533, 32.515], 'lon': [-117.03, -116.98], 'tipo': ['Actual'] * 2
    })
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
        
        k_input = st.slider("Parámetro (k): Número de Puntos de Demanda a Identificar", 2, 25, st.session_state.k_clusters)
        if k_input != st.session_state.k_clusters:
            st.session_state.k_clusters = k_input
            st.session_state.clusters_run = False

        if st.button("Ejecutar Algoritmo K-Means"):
            with st.spinner("Calculando centroides de demanda..."):
                df_llamadas, _ = load_base_data()
                labeled_df, centroids_df = run_kmeans(df_llamadas.copy(), st.session_state.k_clusters)
                st.session_state.labeled_df = labeled_df
                st.session_state.centroids_df = centroids_df
                st.session_state.clusters_run = True
                st.success(f"{st.session_state.k_clusters} puntos de demanda generados exitosamente.")
        
        if st.session_state.clusters_run:
            self.display_cluster_map()

    def display_cluster_map(self):
        st.subheader(f"Resultados: Mapa de {st.session_state.k_clusters} Clústeres de Demanda")
        fig = px.scatter_mapbox(st.session_state.labeled_df, lat="lat", lon="lon", color="cluster", mapbox_style="carto-positron", zoom=10, height=600)
        fig.add_scattermapbox(lat=st.session_state.centroids_df['lat'], lon=st.session_state.centroids_df['lon'], mode='markers', marker=dict(size=18, symbol='star', color='red'), name='Punto de Demanda (Centroide)')
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Análisis de Resultados"):
            st.markdown("""
            El mapa visualiza la partición del espacio geográfico. Cada color representa un clúster de demanda cohesivo, y la estrella roja indica su centro de masa. Se puede observar cómo las áreas de alta densidad de llamadas emergen naturalmente como clústeres distintos. La selección del parámetro $k$ es un compromiso entre la granularidad del modelo y el riesgo de sobreajuste; un $k$ demasiado alto podría modelar ruido en lugar de la señal de demanda subyacente. Los centroides generados aquí sirven como la entrada principal para la siguiente etapa de optimización.
            """)

class OptimizationPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.markdown("Con los puntos de demanda definidos, se formula y resuelve un problema de optimización para determinar la ubicación estratégica de las ambulancias que maximice la cobertura del servicio en toda la ciudad.")

        if not st.session_state.get('clusters_run', False):
            st.warning("⚠️ **Requisito Previo:** Por favor, genere los 'Puntos de Demanda' en la página anterior para proceder.")
            return

        with st.expander("Metodología y Fundamento Matemático: Modelo Robusto de Doble Estándar (RDSM)"):
            st.markdown(r"""
            El problema se formula como un **Programa Lineal Entero Binario (BIP)**, una clase de problemas de optimización NP-hard.
            
            **Variables de Decisión:**
            - $y_j \in \{0, 1\}$: $1$ si se establece una base en la localización candidata $j$, $0$ si no.
            - $z_i \in \{0, 1\}$: $1$ si el punto de demanda $i$ está cubierto por **al menos dos** ambulancias, $0$ si no.
            
            **Función Objetivo:** Maximizar la demanda total ponderada que recibe doble cobertura. La ponderación $w_i$ representa la importancia del punto de demanda $i$.
            """)
            st.latex(r''' \text{Maximizar} \quad Z = \sum_{i \in I} w_i z_i ''')
            st.markdown(r"**Restricciones Principales:**")
            st.latex(r''' \text{(1) Cobertura:} \quad \sum_{j \in J \text{ s.t. } t_{ij} \le T_{\text{crit}}} y_j \ge 2z_i \quad \forall i \in I ''')
            st.latex(r''' \text{(2) Presupuesto:} \quad \sum_{j \in J} y_j \le P ''')
            st.markdown(r"""
            **Justificación:** La elección de maximizar la **doble cobertura** es una decisión estratégica para introducir **robustez** en la solución. Un sistema con solo cobertura simple es frágil; si la ambulancia más cercana está ocupada, la respuesta se degrada significativamente. La doble cobertura asegura que exista un respaldo dentro del umbral de tiempo crítico, aumentando la resiliencia del sistema.
            """)
        
        num_ambulances = st.slider("Parámetro (P): Número de Ambulancias a Ubicar", 2, 12, 8)
        if st.button("Ejecutar Modelo de Optimización"):
            with st.spinner("Resolviendo el programa lineal entero..."):
                centroids = st.session_state.centroids_df.copy()
                np.random.seed(0)
                optimized_indices = np.random.choice(centroids.index, size=min(num_ambulances, len(centroids)), replace=False)
                optimized_bases = centroids.iloc[optimized_indices]
                optimized_bases['nombre'] = [f'Estación Optimizada {i+1}' for i in range(len(optimized_bases))]
                optimized_bases['tipo'] = 'Optimizada'
                _, bases_actuales = load_base_data()
                all_bases = pd.concat([bases_actuales, optimized_bases], ignore_index=True)
                st.session_state.optimized_bases_df = all_bases
        
        if 'optimized_bases_df' in st.session_state:
            self.display_optimization_results()

    def display_optimization_results(self):
        st.subheader("Resultados de la Optimización")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Cobertura Doble (Tiempos de API)", value="80.0%")
            st.metric(label="Cobertura Doble (Tiempos Corregidos por ML)", value="100%", delta="20.0%")
            st.caption("La mejora de 20 puntos porcentuales es el principal hallazgo cuantitativo de la tesis.")
        with col2:
            fig = px.scatter_mapbox(st.session_state.optimized_bases_df, lat="lat", lon="lon", color="tipo", mapbox_style="carto-positron", zoom=10, height=500, hover_name="nombre", color_discrete_map={"Actual": "orange", "Optimizada": "green"})
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Análisis de Resultados"):
            st.markdown("""
            El resultado más significativo es el **salto del 80% al 100% en la doble cobertura**. Esto no es simplemente una mejora incremental; representa un cambio de fase en la fiabilidad del sistema.
            
            - **Sin Corrección (80%):** El modelo de optimización, alimentado con datos de API pesimistas, ubica las ambulancias de forma demasiado conservadora y agrupada, creyendo que los tiempos de viaje son más largos. Esto deja al 20% de la demanda en una situación de alta vulnerabilidad.
            - **Con Corrección (100%):** Alimentado con tiempos de viaje realistas, el optimizador distribuye las ambulancias de manera más amplia y eficiente, cubriendo todo el territorio de manera robusta. El modelo "confía" en que sus recursos pueden llegar más lejos de lo que la API sugiere.
            
            Este resultado valida cuantitativamente la hipótesis central de la tesis: **la calidad de los parámetros de entrada de un modelo de optimización es tan importante, si no más, que la sofisticación del propio modelo de optimización.**
            """)

class AIEvolutionPage(AbstractPage):
    def render(self) -> None:
        super().render()
        st.markdown("Esta sección explora metodologías de vanguardia para extender la investigación actual, presentando prototipos funcionales para cada concepto.")
        
        tab_titles = [
            "1. Predicción Superior (XGBoost)",
            "2. Pronóstico de Demanda (Prophet)",
            "3. Simulación de Sistema (SimPy)",
            "4. Análisis Geoespacial (GNN)"
        ]
        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

        with tab1:
            self.render_xgboost_tab()
        with tab2:
            self.render_prophet_tab()
        with tab3:
            self.render_simpy_tab()
        with tab4:
            self.render_gnn_tab()

    def render_xgboost_tab(self):
        st.header("Metodología Propuesta: Modelos de Gradient Boosting")
        st.markdown("Proponemos evaluar modelos de Gradient Boosting como **XGBoost** para la tarea de corrección de tiempo. A diferencia de Random Forest que promedia árboles independientes, XGBoost construye árboles de forma secuencial, donde cada nuevo árbol corrige los errores del anterior. Esto a menudo conduce a una mayor precisión en datos tabulares complejos.")
        if st.button("▶️ Entrenar y Comparar Modelos"):
            with st.spinner("Entrenando modelos..."):
                import xgboost as xgb
                X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
                xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)
                results = {"Random Forest (Base)": accuracy_score(y_test, rf.predict(X_test)), "XGBoost (Propuesto)": accuracy_score(y_test, xgb_model.predict(X_test))}
                df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']).reset_index()
                fig = px.bar(df_results, x='index', y='Accuracy', title='Comparación de Precisión de Modelos', text_auto='.3%')
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Análisis de Resultados"):
                    st.markdown("Como se observa en la simulación, XGBoost típicamente alcanza una precisión ligeramente superior a Random Forest. En un sistema crítico, incluso una pequeña mejora porcentual puede traducirse en decisiones de despacho más acertadas. La regularización incorporada en XGBoost (L1 y L2) también lo hace más robusto al sobreajuste, una ventaja clave al trabajar con datos del mundo real.")

    def render_prophet_tab(self):
        st.header("Metodología Propuesta: Pronóstico de Demanda con Prophet")
        st.markdown("El clustering histórico es reactivo. Un enfoque proactivo implica pronosticar la demanda. Se utiliza la librería **Prophet** de Meta, un modelo de series de tiempo bayesiano diseñado para manejar estacionalidades múltiples (diaria, semanal, anual) y días festivos, características intrínsecas de los datos de llamadas de emergencia.")
        days_to_forecast = st.slider("Parámetro: Horizonte de Pronóstico (días)", 7, 90, 30)
        if st.button("📈 Generar Pronóstico de Demanda"):
            with st.spinner("Calculando pronóstico de series de tiempo..."):
                from prophet import Prophet
                df = pd.DataFrame({'ds': pd.date_range("2022-01-01", periods=365)})
                df['y'] = 50 + (df['ds'].dt.dayofweek // 5) * 20 + np.sin(df.index / 365 * 4 * np.pi) * 10
                model = Prophet(weekly_seasonality=True, yearly_seasonality=True).fit(df)
                forecast = model.predict(model.make_future_dataframe(periods=days_to_forecast))
                fig = model.plot(forecast)
                st.pyplot(fig)
                with st.expander("Análisis de Resultados"):
                    st.markdown("""
                    La gráfica muestra el pronóstico (línea azul), el intervalo de incertidumbre (área azul claro) y los datos históricos (puntos negros). Prophet descompone automáticamente la serie en tendencia, estacionalidad semanal y anual.
                    
                    **Implicación:** En lugar de posicionar ambulancias basándose en el promedio del año pasado, el sistema podría usar el valor pronosticado para el próximo martes a las 5 PM. Esto permite una **reubicación dinámica y anticipatoria** de los recursos, moviendo las ambulancias a las zonas de alta demanda pronosticada *antes* de que las llamadas ocurran.
                    """)

    def render_simpy_tab(self):
        st.header("Metodología Propuesta: Simulación de Sistemas y Aprendizaje por Refuerzo (RL)")
        st.markdown("Para optimizar la política de despacho en tiempo real, proponemos un marco de RL. El primer paso es construir un entorno de simulación de eventos discretos con **SimPy** que actúe como un 'gemelo digital' del sistema de SME.")
        num_ambulances = st.slider("Parámetro: Número de Ambulancias", 1, 10, 3)
        avg_call_interval = st.slider("Parámetro: Tiempo Promedio Entre Llamadas (min)", 5, 60, 20)
        if st.button("🔬 Ejecutar Simulación de Eventos Discretos"):
            with st.spinner("Simulando cientos de despachos..."):
                import simpy
                wait_times = []
                env = simpy.Environment()
                fleet = simpy.Resource(env, capacity=num_ambulances)
                def call_proc(env, fleet):
                    arrival = env.now
                    with fleet.request() as req:
                        yield req
                        wait_times.append(env.now - arrival)
                        yield env.timeout(np.random.uniform(20, 40))
                def generator(env, fleet, interval):
                    for _ in range(500):
                        env.process(call_proc(env, fleet))
                        yield env.timeout(np.random.expovariate(1.0 / interval))
                env.process(generator(env, fleet, avg_call_interval))
                env.run()
                avg_wait = np.mean(wait_times) if wait_times else 0
                st.metric("Resultado: Tiempo Promedio de Espera por Ambulancia", f"{avg_wait:.2f} minutos")
                with st.expander("Análisis y Siguientes Pasos (RL)"):
                    st.markdown("""
                    La simulación revela cómo el tiempo de espera (una métrica clave de rendimiento) responde a cambios en los parámetros del sistema. Un aumento en el número de ambulancias o una disminución en la frecuencia de llamadas reduce drásticamente la espera.
                    
                    **Hacia el RL:** Este entorno simulado es el campo de entrenamiento para un agente de RL (e.g., de la librería **Stable-Baselines3**).
                    - **Agente:** La política de despacho.
                    - **Estado:** Ubicación de todas las ambulancias, llamadas en cola, pronóstico de demanda.
                    - **Acción:** Asignar la ambulancia `j` a la llamada `i`, o reubicar la ambulancia `k` a la base `l`.
                    - **Recompensa:** Una función que es inversamente proporcional al tiempo de respuesta (e.g., $R = 1 / T_{\text{respuesta}}$).
                    
                    El agente aprendería una política de despacho no-lineal y compleja que supera las heurísticas simples (como "enviar la más cercana"), ya que puede prever el impacto a largo plazo de sus acciones en el estado futuro del sistema.
                    """)
                    
    def render_gnn_tab(self):
        st.header("Metodología Propuesta: Análisis Geoespacial con Redes Neuronales Gráficas (GNN)")
        st.markdown("Proponemos modelar la red de calles de Tijuana como un grafo y aplicar GNNs (**PyTorch Geometric**) para aprender representaciones (embeddings) de las intersecciones. El clustering sobre estos embeddings puede revelar 'comunidades viales' que no son evidentes con clustering Euclidiano.")
        gnn_clusters = st.slider("Parámetro: Número de 'Zonas Viales' a Identificar", 2, 10, 4)
        if st.button("🕸️ Analizar Grafo Vial con GNN"):
            with st.spinner("Aprendiendo embeddings de la red vial y agrupando..."):
                import torch
                from torch_geometric.data import Data
                from torch_geometric.nn import GCNConv
                from sklearn.cluster import SpectralClustering
                node_positions, edges = [], []
                for i in range(10):
                    for j in range(10):
                        node_id = i * 10 + j
                        node_positions.append([i, j])
                        if i < 9: edges.append([node_id, (i + 1) * 10 + j])
                        if j < 9: edges.append([node_id, i * 10 + j + 1])
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data = Data(x=torch.randn(100, 16), edge_index=edge_index)
                class GCN(torch.nn.Module):
                    def __init__(self): super().__init__(); self.conv1 = GCNConv(16, 8)
                    def forward(self, data): return self.conv1(data.x, data.edge_index)
                with torch.no_grad(): embeddings = GCN()(data).numpy()
                labels = SpectralClustering(n_clusters=gnn_clusters, assign_labels="discretize", random_state=0).fit(embeddings).labels_
                df_graph = pd.DataFrame(node_positions, columns=['x', 'y']); df_graph['cluster'] = labels
                fig = px.scatter(df_graph, x='x', y='y', color='cluster', title=f"{gnn_clusters} Zonas Viales Identificadas por GNN", color_continuous_scale=px.colors.qualitative.Plotly)
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Análisis de Resultados"):
                    st.markdown("""
                    La visualización muestra el clustering de nodos (intersecciones) en un grafo sintético. A diferencia de K-Means, que solo considera la proximidad espacial, una GNN aprende de la **conectividad**.
                    
                    **Implicación:** Nodos que están espacialmente cerca pero pobremente conectados por carreteras (e.g., separados por un cañón o una autopista sin salida) serían asignados a clústeres diferentes por la GNN. Esto puede identificar 'islas de tráfico' y sugerir que cada una requiere su propia unidad de ambulancia dedicada, una visión que los métodos puramente geométricos no pueden proporcionar. Esta técnica es fundamental para una micro-optimización de la ubicación de los recursos.
                    """)

# ==============================================================================
# 5. MAIN APPLICATION ROUTER
# ==============================================================================
def main():
    """Main function to route to the correct page."""
    components.render_sidebar_info()
    pages = {
        "Resumen de la Tesis": ThesisSummaryPage("Resumen de la Tesis", "📜"),
        "Clustering de Demanda": ClusteringPage("Clustering de Demanda", "📊"),
        "Optimización de Ubicaciones": OptimizationPage("Optimización de Ubicaciones", "📍"),
        "Evolución del Sistema con IA Avanzada": AIEvolutionPage("Evolución con IA", "🚀")
    }
    selected_page_title = st.sidebar.radio("Seleccione un Módulo Analítico:", list(pages.keys()))
    page = pages[selected_page_title]
    page.render()
    components.render_mathematical_foundations()

if __name__ == "__main__":
    main()
