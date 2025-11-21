# Student Performance Prediction System

Dashboard interactivo para análisis y predicción del rendimiento académico de estudiantes utilizando Machine Learning y Deep Learning.

## 📊 Descripción del Proyecto

Sistema completo de predicción del rendimiento estudiantil que integra:
- Base de datos en la nube (Azure Cosmos DB con API MongoDB)
- Tres modelos de Machine Learning/Deep Learning
- Dashboard interactivo con Streamlit
- Visualizaciones dinámicas con Plotly

**Proyecto desarrollado para:** Big Data Analytics - Universidad del Norte

## 🚀 Demo en Vivo

[Ver Dashboard en Vivo](https://tu-usuario-student-performance-dashboard.streamlit.app)

## 🎯 Características Principales

### Modelos Implementados
1. **Red Neuronal Artificial (ANN)** - TensorFlow/Keras
2. **Random Forest** - Scikit-learn
3. **XGBoost** - XGBoost Library

### Funcionalidades del Dashboard
- **Página de Inicio:** Métricas generales y estadísticas de la base de datos
- **Exploración de Datos:** Visualizaciones interactivas, correlaciones y distribuciones
- **Predicciones:** Interfaz para predecir GPA con los 3 modelos simultáneamente
- **Comparación de Modelos:** Métricas de evaluación y rendimiento
- **Análisis Estadístico:** Análisis univariado, bivariado y multivariado

## 🛠️ Tecnologías Utilizadas

- **Frontend:** Streamlit
- **Backend:** Python 3.12
- **Base de Datos:** Azure Cosmos DB (MongoDB API)
- **ML/DL:** TensorFlow, Scikit-learn, XGBoost
- **Visualización:** Plotly
- **Deployment:** Streamlit Community Cloud

## 📦 Instalación Local

### Requisitos Previos
- Python 3.8 o superior
- Acceso a Azure Cosmos DB

### Pasos de Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/student-performance-dashboard.git
cd student-performance-dashboard
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Ejecutar la aplicación:
```bash
streamlit run app.py
```

5. Abrir en navegador:
```
http://localhost:8501
```

## 📁 Estructura del Proyecto

```
student-performance-dashboard/
├── app.py                                  # Aplicación principal
├── requirements.txt                        # Dependencias
├── student_performance_neural_network.h5   # Modelo Red Neuronal
├── scaler_neural_network.pkl              # Scaler para normalización
├── random_forest_model.pkl                # Modelo Random Forest
├── xgboost_model.pkl                      # Modelo XGBoost
├── model_results.json                     # Métricas de modelos
├── regenerar_todos_modelos.py             # Script de reentrenamiento
├── README.md                              # Este archivo
├── GUIA_DEPLOYMENT.txt                    # Guía de deployment
└── .streamlit/
    └── config.toml                        # Configuración Streamlit
```

## 📊 Variables del Dataset

El sistema analiza 30+ variables incluyendo:
- **Demográficas:** Edad, Género, Dirección
- **Académicas:** Calificaciones previas, Horas de estudio, Asistencia, Fracasos
- **Familiares:** Educación parental, Tamaño de familia, Apoyo familiar
- **Actividades:** Extracurriculares, Deportes, Tutorías
- **Salud y Hábitos:** Horas de sueño, Estado de salud, Consumo de alcohol
- **Objetivo:** GPA (Grade Point Average)

## 🎓 Metodología

### Preprocesamiento
- Codificación de variables categóricas con LabelEncoder
- Normalización con StandardScaler
- División 80/20 (entrenamiento/prueba)

### Entrenamiento de Modelos
- **Red Neuronal:** 
  - Arquitectura: 29 → 128 → 64 → 32 → 1
  - Dropout para prevenir overfitting
  - Early stopping basado en validation loss
  
- **Random Forest:**
  - 100 árboles
  - Profundidad máxima: 10
  
- **XGBoost:**
  - 100 estimadores
  - Learning rate: 0.1
  - Profundidad máxima: 6

### Métricas de Evaluación
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)  
- RMSE (Root Mean Squared Error)
- R² Score

## 📈 Resultados

Los modelos muestran un alto rendimiento en la predicción del GPA:
- R² > 0.90 en los tres modelos
- MAE < 0.15 puntos de GPA
- Capacidad de predicción confiable para intervención académica temprana

## 🔒 Seguridad

- Credenciales de MongoDB almacenadas en Streamlit Secrets
- No se exponen datos sensibles en el repositorio público
- Conexión segura con Azure Cosmos DB mediante TLS

## 👥 Equipo

**Desarrollador:** Shirley P.  
**Institución:** Universidad del Norte  
**Programa:** Big Data Analytics  
**Fecha:** Noviembre 2024

## 📄 Licencia

Este proyecto es parte de un trabajo académico y está disponible solo para fines educativos.

## 🤝 Contribuciones

Este es un proyecto académico individual. No se aceptan contribuciones externas.

## 📞 Contacto

Para preguntas sobre el proyecto:
- Email: [tu email]
- Universidad del Norte

## 🙏 Agradecimientos

- Universidad del Norte - Departamento de Ingeniería
- Profesor del curso de Big Data Analytics
- Microsoft Azure por el crédito educativo
- Comunidad de Streamlit

---

**Nota:** Este dashboard fue desarrollado como parte del Proyecto 2: Cloud Document Database with Predictive Analytics para el curso de Big Data Analytics.
