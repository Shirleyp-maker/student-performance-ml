import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
import tensorflow as tf
import pickle
from datetime import datetime
import json

# Configuración de la página
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2c3e50;
    }
    h3 {
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

# Función para conectar a MongoDB
@st.cache_resource
def get_mongo_connection():
    """Establece conexión con MongoDB Azure Cosmos DB"""
    connection_string = "mongodb+srv://shirleyp:Bigdata2$@student-performance-mongo.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false"
    try:
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        client.server_info()
        return client
    except Exception as e:
        st.error(f"Error conectando a MongoDB: {str(e)}")
        return None

# Función para cargar modelos
@st.cache_resource
def load_models():
    """Carga los modelos entrenados"""
    nn_model = None
    scaler = None
    rf_model = None
    xgb_model = None
    
    # Intentar cargar Red Neuronal
    try:
        nn_model = tf.keras.models.load_model('student_performance_neural_network.h5', compile=False)
        nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        st.success("Red Neuronal cargada correctamente")
    except Exception as e:
        st.warning(f"No se pudo cargar Red Neuronal: {str(e)}")
    
    # Intentar cargar Scaler
    try:
        import pickle5 as pickle_load
    except:
        import pickle as pickle_load
    
    try:
        with open('scaler_neural_network.pkl', 'rb') as f:
            scaler = pickle_load.load(f)
        st.success("Scaler cargado correctamente")
    except Exception as e:
        st.warning(f"No se pudo cargar Scaler: {str(e)}")
    
    # Intentar cargar Random Forest
    try:
        with open('random_forest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        st.success("Random Forest cargado correctamente")
    except Exception as e:
        st.warning(f"No se pudo cargar Random Forest: {str(e)}")
    
    # Intentar cargar XGBoost
    try:
        with open('xgboost_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        st.success("XGBoost cargado correctamente")
    except Exception as e:
        st.warning(f"No se pudo cargar XGBoost: {str(e)}")
    
    return nn_model, scaler, rf_model, xgb_model

# Función para cargar datos desde MongoDB
@st.cache_data(ttl=600)
def load_data_from_mongo(_client, database_name, collection_name):
    """Carga datos desde MongoDB"""
    try:
        db = _client[database_name]
        collection = db[collection_name]
        data = list(collection.find())
        df = pd.DataFrame(data)
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return pd.DataFrame()

# Función principal
def main():
    st.title("Student Performance Prediction System")
    st.markdown("### Dashboard de Análisis y Predicción del Rendimiento Estudiantil")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navegación")
    page = st.sidebar.radio(
        "Seleccione una sección:",
        ["Inicio", "Exploración de Datos", "Predicciones", "Comparación de Modelos", "Análisis Estadístico"]
    )
    
    # Conectar a MongoDB
    client = get_mongo_connection()
    if client is None:
        st.error("No se pudo establecer conexión con la base de datos")
        return
    
    # Cargar modelos
    nn_model, scaler, rf_model, xgb_model = load_models()
    
    models_available = any([nn_model is not None, rf_model is not None, xgb_model is not None])
    if not models_available:
        st.error("No se pudo cargar ningún modelo. Verifique que los archivos de modelos estén en el directorio correcto.")
    
    # Página de Inicio
    if page == "Inicio":
        show_home_page(client)
    
    # Página de Exploración de Datos
    elif page == "Exploración de Datos":
        show_data_exploration(client)
    
    # Página de Predicciones
    elif page == "Predicciones":
        show_predictions(nn_model, scaler, rf_model, xgb_model)
    
    # Página de Comparación de Modelos
    elif page == "Comparación de Modelos":
        show_model_comparison()
    
    # Página de Análisis Estadístico
    elif page == "Análisis Estadístico":
        show_statistical_analysis(client)

def show_home_page(client):
    """Muestra la página de inicio con resumen del proyecto"""
    st.header("Bienvenido al Sistema de Predicción de Rendimiento Estudiantil")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Base de Datos",
            value="Azure Cosmos DB",
            delta="MongoDB API"
        )
    
    with col2:
        st.metric(
            label="Modelos Implementados",
            value="3",
            delta="NN, RF, XGBoost"
        )
    
    with col3:
        st.metric(
            label="Variables Analizadas",
            value="10+",
            delta="Predictores"
        )
    
    st.markdown("---")
    
    st.subheader("Descripción del Proyecto")
    st.write("""
    Este sistema utiliza técnicas avanzadas de Machine Learning y Deep Learning para predecir 
    el rendimiento académico de estudiantes basándose en múltiples factores como:
    
    - Horas de estudio
    - Asistencia a clases
    - Participación en actividades extracurriculares
    - Horas de sueño
    - Nivel educativo de los padres
    - Y más...
    
    El proyecto implementa tres modelos de predicción:
    1. Red Neuronal Artificial (ANN)
    2. Random Forest
    3. XGBoost
    """)
    
    st.markdown("---")
    
    st.subheader("Estadísticas de la Base de Datos")
    
    try:
        # Cargar datos para mostrar estadísticas
        df = load_data_from_mongo(client, "student_performance", "students")
        
        if not df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de Estudiantes", len(df))
            
            with col2:
                if 'GPA' in df.columns:
                    st.metric("GPA Promedio", f"{df['GPA'].mean():.2f}")
            
            with col3:
                if 'StudyTimeWeekly' in df.columns:
                    st.metric("Horas de Estudio Promedio", f"{df['StudyTimeWeekly'].mean():.1f}")
            
            with col4:
                if 'Attendance' in df.columns:
                    st.metric("Asistencia Promedio", f"{df['Attendance'].mean():.1f}%")
    except Exception as e:
        st.warning("No se pudieron cargar las estadísticas de la base de datos")

def show_data_exploration(client):
    """Muestra la página de exploración de datos"""
    st.header("Exploración de Datos")
    
    # Cargar datos
    df = load_data_from_mongo(client, "student_performance", "students")
    
    if df.empty:
        st.warning("No hay datos disponibles en la base de datos")
        return
    
    st.subheader("Vista Previa de los Datos")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(df.describe(), use_container_width=True)
    
    with col2:
        st.subheader("Información del Dataset")
        info_data = {
            "Columna": df.columns,
            "Tipo de Dato": [str(dtype) for dtype in df.dtypes],
            "Valores No Nulos": [df[col].count() for col in df.columns],
            "Valores Nulos": [df[col].isnull().sum() for col in df.columns]
        }
        st.dataframe(pd.DataFrame(info_data), use_container_width=True)
    
    st.markdown("---")
    
    # Visualizaciones
    st.subheader("Visualizaciones Interactivas")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Distribuciones", "Correlaciones", "Comparaciones", "Tendencias"])
    
    with tab1:
        st.write("Distribución de Variables Numéricas")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Seleccione una variable:", numeric_cols, key="dist_col")
            fig = px.histogram(df, x=selected_col, nbins=30, title=f"Distribución de {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.write("Matriz de Correlación")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, 
                          text_auto='.2f',
                          aspect="auto",
                          title="Matriz de Correlación",
                          color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.write("Comparación entre Variables")
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Variable X:", numeric_cols, key="x_var")
        with col2:
            y_var = st.selectbox("Variable Y:", numeric_cols, key="y_var")
        
        fig = px.scatter(df, x=x_var, y=y_var, 
                        title=f"{x_var} vs {y_var}",
                        trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        if 'GPA' in df.columns:
            st.write("Análisis de GPA")
            fig = px.box(df, y='GPA', title="Distribución de GPA")
            st.plotly_chart(fig, use_container_width=True)

def show_predictions(nn_model, scaler, rf_model, xgb_model):
    """Muestra la página de predicciones"""
    st.header("Predicción de Rendimiento Estudiantil")
    
    if rf_model is None and xgb_model is None and nn_model is None:
        st.error("Ningún modelo está disponible para realizar predicciones")
        return
    
    st.write("Ingrese los datos del estudiante para predecir su GPA:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Información Personal")
        age = st.slider("Edad", 15, 22, 17)
        gender = st.selectbox("Género", ["Male", "Female"])
        address = st.selectbox("Tipo de Dirección", ["Urban", "Rural"])
        
        st.subheader("Estudios")
        previous_grades = st.slider("Calificaciones Previas (0-20)", 0, 20, 15)
        study_hours = st.slider("Horas de Estudio Semanales", 0, 40, 10)
        failures = st.slider("Número de Fracasos Previos", 0, 4, 0)
    
    with col2:
        st.subheader("Familia")
        mother_ed = st.selectbox("Educación Madre", ["None", "Primary", "Secondary", "Higher"])
        father_ed = st.selectbox("Educación Padre", ["None", "Primary", "Secondary", "Higher"])
        family_size = st.selectbox("Tamaño Familia", ["LE3", "GT3"])
        parent_status = st.selectbox("Estado Padres", ["Together", "Apart"])
        parental_involvement = st.selectbox("Involucramiento Parental", ["Low", "Medium", "High"])
        
    with col3:
        st.subheader("Actividades y Apoyo")
        extracurricular = st.selectbox("Actividades Extracurriculares", ["No", "Yes"])
        school_support = st.selectbox("Apoyo Escolar", ["No", "Yes"])
        family_support = st.selectbox("Apoyo Familiar", ["No", "Yes"])
        paid_classes = st.selectbox("Clases Pagadas", ["No", "Yes"])
        tutoring = st.slider("Sesiones de Tutoría/mes", 0, 10, 2)
        
        st.subheader("Otros")
        travel_time = st.slider("Tiempo de Viaje (min)", 15, 120, 30)
        free_time = st.slider("Tiempo Libre (horas/día)", 1, 8, 4)
        going_out = st.slider("Salidas Sociales (1-5)", 1, 5, 3)
        health = st.slider("Estado de Salud (1-5)", 1, 5, 4)
        sleep_hours = st.slider("Horas de Sueño", 4, 12, 7)
        
    col1_extra, col2_extra, col3_extra = st.columns(3)
    
    with col1_extra:
        higher_ed = st.selectbox("¿Quiere Educación Superior?", ["No", "Yes"])
        internet = st.selectbox("Acceso a Internet", ["No", "Yes"])
    
    with col2_extra:
        school = st.selectbox("Escuela", ["GP", "MS"])
        reason = st.selectbox("Razón de Elección", ["Home", "Reputation", "Course", "Other"])
    
    with col3_extra:
        guardian = st.selectbox("Tutor", ["Mother", "Father", "Other"])
        absences = st.slider("Ausencias", 0, 50, 5)
        alcohol = st.slider("Consumo Alcohol (1-5)", 1, 5, 1)
        attendance = st.slider("Asistencia (%)", 0, 100, 85)
    
    # Preparar datos para predicción
    if st.button("Realizar Predicción", type="primary"):
        # Mapeo de valores categóricos a numéricos (según LabelEncoder)
        gender_map = {"Male": 1, "Female": 0}
        address_map = {"Urban": 1, "Rural": 0}
        yes_no_map = {"Yes": 1, "No": 0}
        education_map = {"None": 0, "Primary": 1, "Secondary": 2, "Higher": 3}
        family_size_map = {"LE3": 0, "GT3": 1}
        parent_status_map = {"Together": 1, "Apart": 0}
        involvement_map = {"Low": 0, "Medium": 1, "High": 2}
        school_map = {"GP": 0, "MS": 1}
        reason_map = {"Home": 0, "Reputation": 1, "Course": 2, "Other": 3}
        guardian_map = {"Mother": 0, "Father": 1, "Other": 2}
        
        # Crear diccionario de datos con el orden correcto
        input_data = {
            'Age': age,
            'Gender': gender_map[gender],
            'Address': address_map[address],
            'Previous_Grades': previous_grades,
            'Study_Hours_Per_Week': study_hours,
            'Attendance': attendance,
            'Failures': failures,
            'Mother_Education': education_map[mother_ed],
            'Father_Education': education_map[father_ed],
            'Family_Size': family_size_map[family_size],
            'Parent_Status': parent_status_map[parent_status],
            'Parental_Involvement': involvement_map[parental_involvement],
            'Extracurricular_Activities': yes_no_map[extracurricular],
            'School_Support': yes_no_map[school_support],
            'Family_Support': yes_no_map[family_support],
            'Paid_Classes': yes_no_map[paid_classes],
            'Tutoring_Sessions': tutoring,
            'Travel_Time': travel_time,
            'Free_Time': free_time,
            'Going_Out': going_out,
            'Health': health,
            'Sleep_Hours': sleep_hours,
            'Wants_Higher_Ed': yes_no_map[higher_ed],
            'Internet_Access': yes_no_map[internet],
            'School': school_map[school],
            'Reason_Choice': reason_map[reason],
            'Guardian': guardian_map[guardian],
            'Absences': absences,
            'Alcohol_Consumption': alcohol
        }
        
        # Convertir a DataFrame
        input_df = pd.DataFrame([input_data])
        
        predictions = {}
        
        # Realizar predicción con Red Neuronal
        if nn_model is not None and scaler is not None:
            try:
                input_scaled = scaler.transform(input_df)
                nn_pred = nn_model.predict(input_scaled, verbose=0)[0][0]
                predictions['Red Neuronal'] = float(nn_pred)
            except Exception as e:
                st.warning(f"Error en predicción con Red Neuronal: {str(e)}")
        
        # Realizar predicción con Random Forest
        if rf_model is not None:
            try:
                rf_pred = rf_model.predict(input_df)[0]
                predictions['Random Forest'] = float(rf_pred)
            except Exception as e:
                st.warning(f"Error en predicción con Random Forest: {str(e)}")
        
        # Realizar predicción con XGBoost
        if xgb_model is not None:
            try:
                xgb_pred = xgb_model.predict(input_df)[0]
                predictions['XGBoost'] = float(xgb_pred)
            except Exception as e:
                st.warning(f"Error en predicción con XGBoost: {str(e)}")
        
        if not predictions:
            st.error("No se pudieron realizar predicciones con ningún modelo")
            return
        
        # Mostrar resultados
        st.markdown("---")
        st.subheader("Resultados de las Predicciones")
        
        cols = st.columns(len(predictions))
        for idx, (model_name, pred_value) in enumerate(predictions.items()):
            with cols[idx]:
                st.metric(model_name, f"{pred_value:.2f}")
        
        # Promedio de predicciones
        avg_pred = np.mean(list(predictions.values()))
        st.markdown("---")
        st.success(f"GPA Promedio Predicho: {avg_pred:.2f}")
        
        # Interpretación
        if avg_pred >= 3.5:
            interpretation = "Excelente rendimiento académico esperado"
        elif avg_pred >= 3.0:
            interpretation = "Buen rendimiento académico esperado"
        elif avg_pred >= 2.5:
            interpretation = "Rendimiento académico promedio"
        else:
            interpretation = "Se recomienda apoyo académico adicional"
        
        st.info(interpretation)
        
        # Visualización comparativa
        fig = go.Figure(data=[
            go.Bar(name='Modelos', 
                   x=list(predictions.keys()) + ['Promedio'],
                   y=list(predictions.values()) + [avg_pred],
                   marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(predictions)+1])
        ])
        fig.update_layout(
            title="Comparación de Predicciones",
            yaxis_title="GPA Predicho",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

def show_model_comparison():
    """Muestra la comparación de modelos"""
    st.header("Comparación de Rendimiento de Modelos")
    
    # Cargar resultados si existen
    try:
        with open('model_results.json', 'r') as f:
            results = json.load(f)
        
        st.subheader("Métricas de Evaluación")
        
        # Crear DataFrame con resultados
        metrics_data = {
            'Modelo': ['Red Neuronal', 'Random Forest', 'XGBoost'],
            'MAE': [
                results.get('neural_network', {}).get('mae', 0),
                results.get('random_forest', {}).get('mae', 0),
                results.get('xgboost', {}).get('mae', 0)
            ],
            'MSE': [
                results.get('neural_network', {}).get('mse', 0),
                results.get('random_forest', {}).get('mse', 0),
                results.get('xgboost', {}).get('mse', 0)
            ],
            'RMSE': [
                results.get('neural_network', {}).get('rmse', 0),
                results.get('random_forest', {}).get('rmse', 0),
                results.get('xgboost', {}).get('rmse', 0)
            ],
            'R2 Score': [
                results.get('neural_network', {}).get('r2', 0),
                results.get('random_forest', {}).get('r2', 0),
                results.get('xgboost', {}).get('r2', 0)
            ]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)
        
        # Gráficos de comparación
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(df_metrics, x='Modelo', y='MAE', 
                         title='Mean Absolute Error (MAE)',
                         color='Modelo')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(df_metrics, x='Modelo', y='R2 Score', 
                         title='R2 Score',
                         color='Modelo')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Mejor modelo
        best_model_idx = df_metrics['R2 Score'].idxmax()
        best_model = df_metrics.loc[best_model_idx, 'Modelo']
        st.success(f"Mejor Modelo: {best_model}")
        
    except FileNotFoundError:
        st.warning("No se encontraron resultados de modelos. Asegúrese de que el archivo 'model_results.json' existe.")

def show_statistical_analysis(client):
    """Muestra análisis estadístico detallado"""
    st.header("Análisis Estadístico Avanzado")
    
    df = load_data_from_mongo(client, "student_performance", "students")
    
    if df.empty:
        st.warning("No hay datos disponibles")
        return
    
    tab1, tab2, tab3 = st.tabs(["Análisis Univariado", "Análisis Bivariado", "Análisis Multivariado"])
    
    with tab1:
        st.subheader("Análisis Univariado")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_var = st.selectbox("Seleccione variable:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Estadísticas:")
                stats = df[selected_var].describe()
                st.dataframe(stats)
            
            with col2:
                fig = px.box(df, y=selected_var, title=f"Box Plot - {selected_var}")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Análisis Bivariado")
        
        if 'GPA' in df.columns and 'StudyTimeWeekly' in df.columns:
            fig = px.scatter(df, x='StudyTimeWeekly', y='GPA',
                           trendline='ols',
                           title='Relación entre Horas de Estudio y GPA')
            st.plotly_chart(fig, use_container_width=True)
            
            correlation = df['StudyTimeWeekly'].corr(df['GPA'])
            st.metric("Correlación", f"{correlation:.3f}")
    
    with tab3:
        st.subheader("Análisis Multivariado")
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) >= 3:
            cols = st.multiselect("Seleccione variables:", numeric_df.columns.tolist(), default=numeric_df.columns.tolist()[:3])
            
            if len(cols) >= 2:
                fig = px.scatter_matrix(df[cols], title="Matriz de Dispersión")
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
