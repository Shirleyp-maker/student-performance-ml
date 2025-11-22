import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Rendimiento Estudiantil",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    h1 { color: #1f77b4; padding-bottom: 20px; }
    h2 { color: #2c3e50; padding-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Conexión a MongoDB
@st.cache_resource
def get_mongo_connection():
    try:
        connection_string = "mongodb+srv://shirleyp:Bigdata2$@student-performance-mongo.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false"
        client = MongoClient(connection_string)
        db = client['student_performance']
        return db
    except Exception as e:
        st.error(f"Error conectando a MongoDB: {str(e)}")
        return None

@st.cache_data(ttl=600)
def load_data_from_mongo():
    db = get_mongo_connection()
    if db is not None:
        try:
            collection = db['students']
            data = list(collection.find({}, {'_id': 0}))
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            st.error(f"Error cargando datos: {str(e)}")
            return None
    return None

def main():
    st.title("Sistema de Predicción de Rendimiento Estudiantil")
    st.markdown("### Análisis y Predicción basado en Machine Learning")
    st.markdown("---")
    
    with st.spinner('Cargando datos desde MongoDB Azure...'):
        df = load_data_from_mongo()
    
    if df is None or df.empty:
        st.error("No se pudieron cargar los datos. Por favor verifica la conexión a MongoDB.")
        return
    
    # Sidebar
    st.sidebar.header("Filtros y Configuración")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Información del Dataset")
    st.sidebar.info(f"**Total de estudiantes:** {len(df)}")
    st.sidebar.info(f"**Variables:** {len(df.columns)}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Vista General", 
        "Análisis Exploratorio", 
        "Predictor", 
        "Datos"
    ])
    
    with tab1:
        st.header("Resumen Ejecutivo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("GPA Promedio", f"{df['GPA'].mean():.2f}")
        with col2:
            st.metric("Horas de Estudio", f"{df['Study_Hours_Per_Week'].mean():.1f}")
        with col3:
            st.metric("Asistencia", f"{df['Attendance'].mean():.1f}%")
        with col4:
            high_perf = len(df[df['GPA'] >= 3.5])
            st.metric("Alto Rendimiento", f"{high_perf}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='GPA', nbins=30, title='Distribución de GPA')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
           fig = px.scatter(df, x='Study_Hours_Per_Week', y='GPA', 
               title='Horas de Estudio vs GPA')
    
    with tab2:
        st.header("Análisis Exploratorio")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto='.2f', 
                       title='Matriz de Correlación',
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Predictor de Rendimiento Estudiantil")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Información Académica")
            study_hours = st.slider("Horas de Estudio/Semana", 0, 40, 15)
            attendance = st.slider("Asistencia (%)", 0, 100, 85)
            previous_grades = st.slider("Calificaciones Anteriores", 0.0, 4.0, 3.0, 0.1)
        
        with col2:
            st.subheader("Información Personal")
            age = st.slider("Edad", 15, 25, 18)
            parental = st.selectbox("Involucramiento Parental", ['Low', 'Medium', 'High'])
        
        with col3:
            st.subheader("Actividades")
            extracurricular = st.selectbox("Actividades Extracurriculares", ['No', 'Yes'])
            tutoring = st.selectbox("Tutorías", ['No', 'Yes'])
            sleep_hours = st.slider("Horas de Sueño", 4, 12, 7)
        
        if st.button("Predecir GPA", type="primary", use_container_width=True):
            predicted_gpa = 2.0
            predicted_gpa += (study_hours / 40) * 1.2
            predicted_gpa += (attendance / 100) * 0.8
            predicted_gpa += (previous_grades / 4.0) * 0.5
            predicted_gpa += 0.2 if parental == 'High' else (0.1 if parental == 'Medium' else 0)
            predicted_gpa += 0.15 if extracurricular == 'Yes' else 0
            predicted_gpa += 0.15 if tutoring == 'Yes' else 0
            predicted_gpa += (sleep_hours / 12) * 0.2
            predicted_gpa = max(0, min(4.0, predicted_gpa))
            
            st.markdown("---")
            st.subheader("Resultado")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_gpa,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "GPA Predicho"},
                gauge={
                    'axis': {'range': [None, 4.0]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 2.0], 'color': '#ffcccc'},
                        {'range': [2.0, 3.0], 'color': '#ffffcc'},
                        {'range': [3.0, 4.0], 'color': '#ccffcc'}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            if predicted_gpa >= 3.5:
                st.success("**Excelente desempeño esperado!**")
            elif predicted_gpa >= 3.0:
                st.info("**Buen desempeño esperado.**")
            elif predicted_gpa >= 2.5:
                st.warning("**Desempeño moderado.**")
            else:
                st.error("**Requiere apoyo adicional.**")
    
    with tab4:
        st.header("Explorador de Datos")
        st.dataframe(df.describe(), use_container_width=True)
        st.dataframe(df.head(20), use_container_width=True)

if __name__ == "__main__":
    main()
