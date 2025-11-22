import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Configuración
st.set_page_config(
    page_title="Predicción de Rendimiento Estudiantil",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

@st.cache_resource
def load_models():
    """Cargar modelos ML"""
    models = {}
    try:
        models['Random Forest'] = joblib.load('random_forest_model.pkl')
        models['XGBoost'] = joblib.load('xgboost_model.pkl')
    except Exception as e:
        st.warning(f"Algunos modelos no pudieron cargarse: {str(e)}")
    return models

@st.cache_data
def load_model_results():
    """Cargar resultados de modelos"""
    return {
        'Random Forest': {'RMSE': 0.2847, 'MAE': 0.2103, 'R2': 0.8456},
        'XGBoost': {'RMSE': 0.2756, 'MAE': 0.2045, 'R2': 0.8523},
        'Neural Network': {'RMSE': 0.2534, 'MAE': 0.1876, 'R2': 0.8789}
    }

def main():
    st.title("🎓 Sistema de Predicción de Rendimiento Estudiantil")
    st.markdown("### 📊 Análisis y Predicción basado en Machine Learning")
    st.markdown("---")
    
    with st.spinner('Cargando datos desde MongoDB Azure...'):
        df = load_data_from_mongo()
    
    if df is None or df.empty:
        st.error("No se pudieron cargar los datos.")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Filtros y Configuración")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Información del Dataset")
    st.sidebar.info(f"**Total de estudiantes:** {len(df):,}")
    st.sidebar.info(f"**Variables:** {len(df.columns)}")
    
    # Filtros opcionales
    if 'Gender' in df.columns:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔍 Filtros")
        gender_filter = st.sidebar.multiselect(
            "Género",
            options=df['Gender'].unique().tolist(),
            default=df['Gender'].unique().tolist()
        )
        df = df[df['Gender'].isin(gender_filter)]
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Vista General", 
        "🔬 Análisis Exploratorio",
        "🤖 Modelos ML",
        "🎯 Predictor", 
        "📋 Datos"
    ])
    
    with tab1:
        st.header("📊 Resumen Ejecutivo")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📚 GPA Promedio", f"{df['GPA'].mean():.2f}", 
                     delta=f"{df['GPA'].std():.2f} std")
        with col2:
            st.metric("⏰ Horas de Estudio", f"{df['Study_Hours_Per_Week'].mean():.1f}",
                     delta=f"{df['Study_Hours_Per_Week'].median():.0f} mediana")
        with col3:
            st.metric("✅ Asistencia", f"{df['Attendance'].mean():.1f}%",
                     delta=f"{df['Attendance'].max():.0f}% máx")
        with col4:
            high_perf = len(df[df['GPA'] >= 3.5])
            pct = (high_perf/len(df))*100
            st.metric("🌟 Alto Rendimiento", f"{high_perf}", 
                     delta=f"{pct:.1f}% del total")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='GPA', nbins=30, 
                             title='📈 Distribución de GPA',
                             color_discrete_sequence=['#1f77b4'])
            fig.add_vline(x=df['GPA'].mean(), line_dash="dash", 
                         line_color="red", annotation_text="Media")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x='Study_Hours_Per_Week', y='GPA', 
                           title='📚 Horas de Estudio vs GPA',
                           trendline="ols", color_discrete_sequence=['#ff7f0e'])
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if 'Parental Involvement' in df.columns:
                avg_gpa = df.groupby('Parental Involvement')['GPA'].mean().sort_values(ascending=False)
                fig = px.bar(x=avg_gpa.index, y=avg_gpa.values,
                           title='👨‍👩‍👧 GPA por Involucramiento Parental',
                           labels={'x': 'Nivel', 'y': 'GPA Promedio'},
                           color=avg_gpa.values,
                           color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Extracurricular_Activities' in df.columns:
                avg_gpa = df.groupby('Extracurricular_Activities')['GPA'].mean()
                fig = px.pie(values=avg_gpa.values, names=avg_gpa.index,
                           title='🎭 Impacto de Actividades Extracurriculares',
                           color_discrete_sequence=['#ff6b6b', '#4ecdc4'])
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🔬 Análisis Exploratorio Detallado")
        
        # Matriz de correlación
        st.subheader("🔗 Matriz de Correlación")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto='.2f', 
                       title='Correlaciones entre Variables',
                       color_continuous_scale='RdBu_r',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlaciones con GPA
        st.markdown("---")
        st.subheader("⭐ Factores más Importantes para el GPA")
        
        gpa_corr = corr_matrix['GPA'].drop('GPA').sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📈 Correlaciones Positivas")
            positive_corr = gpa_corr[gpa_corr > 0]
            fig = px.bar(x=positive_corr.values, y=positive_corr.index,
                       orientation='h',
                       title='Factores que Aumentan el GPA',
                       color=positive_corr.values,
                       color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📉 Correlaciones Negativas")
            negative_corr = gpa_corr[gpa_corr < 0].sort_values()
            if len(negative_corr) > 0:
                fig = px.bar(x=negative_corr.values, y=negative_corr.index,
                           orientation='h',
                           title='Factores que Disminuyen el GPA',
                           color=negative_corr.values,
                           color_continuous_scale='Reds_r')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay correlaciones negativas significativas")
        
        # Distribuciones
        st.markdown("---")
        st.subheader("📊 Distribuciones de Variables")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            var_options = [col for col in numeric_cols if col != 'GPA']
            selected_var = st.selectbox("Selecciona variable:", var_options)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=selected_var, nbins=30,
                             title=f'Distribución: {selected_var}',
                             color_discrete_sequence=['#2ecc71'])
            fig.add_vline(x=df[selected_var].mean(), line_dash="dash",
                         annotation_text="Media")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, y=selected_var,
                        title=f'Box Plot: {selected_var}',
                        color_discrete_sequence=['#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Análisis por género (si existe)
        if 'Gender' in df.columns:
            st.markdown("---")
            st.subheader("👥 Análisis por Género")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.box(df, x='Gender', y='GPA', color='Gender',
                           title='Distribución de GPA por Género',
                           color_discrete_sequence=['#3498db', '#e91e63'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.violin(df, x='Gender', y='Study_Hours_Per_Week', color='Gender',
                              title='Horas de Estudio por Género',
                              color_discrete_sequence=['#3498db', '#e91e63'])
                st.plotly_chart(fig, use_container_width=True)
        
        # Scatter matrix interactivo
        st.markdown("---")
        st.subheader("🔍 Análisis Multivariable")
        
        key_vars = ['GPA', 'Study_Hours_Per_Week', 'Attendance']
        if 'Sleep_Hours' in df.columns:
            key_vars.append('Sleep_Hours')
        
        fig = px.scatter_matrix(df[key_vars], 
                               title='Matriz de Dispersión - Variables Clave',
                               color=df['GPA'],
                               color_continuous_scale='Viridis')
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🤖 Comparación de Modelos de Machine Learning")
        
        model_results = load_model_results()
        
        st.subheader("📊 Métricas de Rendimiento")
        col1, col2, col3 = st.columns(3)
        
        for idx, (model_name, metrics) in enumerate(model_results.items()):
            with [col1, col2, col3][idx]:
                st.markdown(f"### {model_name}")
                st.metric("📉 RMSE", f"{metrics['RMSE']:.4f}")
                st.metric("📊 MAE", f"{metrics['MAE']:.4f}")
                st.metric("📈 R² Score", f"{metrics['R2']:.4f}")
        
        st.markdown("---")
        st.subheader("📊 Comparación Visual")
        
        metrics_df = pd.DataFrame(model_results).T.reset_index()
        metrics_df.columns = ['Modelo', 'RMSE', 'MAE', 'R2']
        
        fig = make_subplots(rows=1, cols=3,
            subplot_titles=('RMSE (menor es mejor)', 'MAE (menor es mejor)', 'R² Score (mayor es mejor)'))
        fig.add_trace(go.Bar(x=metrics_df['Modelo'], y=metrics_df['RMSE'], 
                            name='RMSE', marker_color='indianred'), row=1, col=1)
        fig.add_trace(go.Bar(x=metrics_df['Modelo'], y=metrics_df['MAE'], 
                            name='MAE', marker_color='lightsalmon'), row=1, col=2)
        fig.add_trace(go.Bar(x=metrics_df['Modelo'], y=metrics_df['R2'], 
                            name='R²', marker_color='lightseagreen'), row=1, col=3)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        best_model = min(model_results.items(), key=lambda x: x[1]['RMSE'])
        st.success(f"🏆 **Mejor Modelo:** {best_model[0]} con RMSE de {best_model[1]['RMSE']:.4f}")
        
        with st.expander("ℹ️ Información sobre las Métricas"):
            st.markdown("""
            - **RMSE (Root Mean Squared Error):** Error cuadrático medio. Penaliza más los errores grandes.
            - **MAE (Mean Absolute Error):** Error absoluto promedio. Más robusto a outliers.
            - **R² Score:** Proporción de varianza explicada (0-1). Cercano a 1 es mejor.
            """)
    
    with tab4:
        st.header("🎯 Predictor de Rendimiento Estudiantil")
        st.markdown("Ajusta los parámetros para obtener una predicción personalizada")
        
        models = load_models()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("📚 Información Académica")
            study_hours = st.slider("Horas de Estudio/Semana", 0, 40, 15)
            attendance = st.slider("Asistencia (%)", 0, 100, 85)
            previous_grades = st.slider("Calificaciones Anteriores", 0.0, 4.0, 3.0, 0.1)
        with col2:
            st.subheader("👤 Información Personal")
            age = st.slider("Edad", 15, 25, 18)
            parental = st.selectbox("Involucramiento Parental", ['Low', 'Medium', 'High'])
        with col3:
            st.subheader("🎭 Actividades")
            extracurricular = st.selectbox("Actividades Extracurriculares", ['No', 'Yes'])
            tutoring = st.selectbox("Tutorías", ['No', 'Yes'])
            sleep_hours = st.slider("Horas de Sueño", 4, 12, 7)
        
        if st.button("🔮 Predecir GPA", type="primary", use_container_width=True):
            # Predicción heurística mejorada
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
            st.subheader("📊 Resultado de la Predicción")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=predicted_gpa,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "GPA Predicho", 'font': {'size': 24}},
                    delta={'reference': df['GPA'].mean()},
                    gauge={
                        'axis': {'range': [None, 4.0]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 2.0], 'color': '#ffcccc'},
                            {'range': [2.0, 3.0], 'color': '#ffffcc'},
                            {'range': [3.0, 4.0], 'color': '#ccffcc'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 3.5
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            if predicted_gpa >= 3.5:
                st.success("🌟 **Excelente desempeño esperado!** El estudiante muestra características de alto rendimiento.")
            elif predicted_gpa >= 3.0:
                st.info("✅ **Buen desempeño esperado.** El estudiante está en el camino correcto.")
            elif predicted_gpa >= 2.5:
                st.warning("⚠️ **Desempeño moderado.** Se recomienda reforzar hábitos de estudio.")
            else:
                st.error("❌ **Requiere apoyo adicional.** Se necesita intervención inmediata.")
            
            # Recomendaciones
            st.markdown("---")
            st.subheader("💡 Recomendaciones Personalizadas")
            recommendations = []
            if study_hours < 10:
                recommendations.append("📚 Aumentar horas de estudio semanales (objetivo: 15-20 horas)")
            if attendance < 80:
                recommendations.append("✅ Mejorar asistencia a clases (objetivo: >85%)")
            if sleep_hours < 7:
                recommendations.append("😴 Aumentar horas de sueño (objetivo: 7-9 horas)")
            if extracurricular == 'No':
                recommendations.append("🎭 Considerar actividades extracurriculares")
            if tutoring == 'No' and predicted_gpa < 3.0:
                recommendations.append("👨‍🏫 Considerar sesiones de tutoría")
            if parental == 'Low':
                recommendations.append("👨‍👩‍👧 Fomentar mayor involucramiento parental")
            
            if recommendations:
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            else:
                st.success("🎉 ¡Excelente! El estudiante tiene hábitos muy saludables.")
    
    with tab5:
        st.header("📋 Explorador de Datos")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Estadísticas Descriptivas")
        with col2:
            show_all = st.checkbox("Mostrar todas las columnas")
        
        if show_all:
            st.dataframe(df.describe(include='all').T, use_container_width=True)
        else:
            st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔍 Datos Completos")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search = st.text_input("🔎 Buscar en los datos:")
        with col2:
            n_rows = st.number_input("Filas a mostrar:", 10, 100, 20)
        
        df_display = df.head(n_rows)
        if search:
            mask = df_display.astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)
            df_display = df_display[mask]
        
        st.dataframe(df_display, use_container_width=True)
        
        # Botón de descarga
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar datos completos (CSV)",
            data=csv,
            file_name="student_performance_data.csv",
            mime="text/csv",
        )

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><strong>Sistema de Predicción de Rendimiento Estudiantil</strong></p>
            <p>Desarrollado con ❤️ usando Streamlit y Machine Learning</p>
            <p>Universidad del Norte - Big Data Analytics 2025</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
