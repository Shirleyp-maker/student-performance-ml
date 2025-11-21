# Dashboard de Predicción de Rendimiento Estudiantil

Sistema de análisis y predicción del rendimiento académico de estudiantes utilizando Machine Learning y Deep Learning con Streamlit.

## Descripción del Proyecto

Este dashboard integra tres modelos de predicción:
- Red Neuronal Artificial (ANN)
- Random Forest
- XGBoost

Conecta con Azure Cosmos DB (MongoDB API) para almacenar y analizar datos de estudiantes.

## Requisitos Previos

- Python 3.8 o superior
- Cuenta de Azure con Cosmos DB configurado
- Modelos entrenados (archivos .h5 y .pkl)

## Instalación

1. Clonar o descargar el repositorio

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Estructura de Archivos Necesaria

Asegúrese de tener los siguientes archivos en el mismo directorio que app.py:

```
proyecto/
├── app.py
├── requirements.txt
├── student_performance_neural_network.h5
├── scaler_neural_network.pkl
├── random_forest_model.pkl
├── xgboost_model.pkl
└── model_results.json (opcional)
```

## Configuración

### Conexión a MongoDB

El dashboard utiliza la siguiente cadena de conexión por defecto:
```
mongodb+srv://shirleyp:Bigdata2$@student-performance-mongo.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false
```

Si necesita cambiar la conexión, edite la función `get_mongo_connection()` en app.py.

### Base de Datos y Colección

Por defecto, el dashboard busca:
- Base de datos: `student_performance`
- Colección: `students`

## Ejecución

Para ejecutar el dashboard:

```bash
streamlit run app.py
```

El dashboard se abrirá automáticamente en su navegador en:
```
http://localhost:8501
```

## Funcionalidades

### 1. Inicio
- Resumen del proyecto
- Métricas generales
- Estadísticas de la base de datos

### 2. Exploración de Datos
- Vista previa de datos
- Estadísticas descriptivas
- Visualizaciones interactivas:
  - Distribuciones de variables
  - Matriz de correlación
  - Comparaciones entre variables
  - Análisis de tendencias

### 3. Predicciones
- Interfaz para ingresar datos de estudiantes
- Predicción con los 3 modelos simultáneamente
- Visualización comparativa de resultados
- Interpretación del GPA predicho

### 4. Comparación de Modelos
- Métricas de evaluación (MAE, MSE, RMSE, R2)
- Gráficos comparativos
- Identificación del mejor modelo

### 5. Análisis Estadístico
- Análisis univariado
- Análisis bivariado
- Análisis multivariado
- Matriz de dispersión

## Variables del Dataset

El sistema analiza las siguientes variables:
- Age: Edad del estudiante
- Gender: Género
- ParentalEducation: Nivel educativo de los padres
- StudyTimeWeekly: Horas de estudio semanales
- Absences: Número de ausencias
- Tutoring: Si recibe tutoría
- ParentalSupport: Nivel de apoyo parental
- Extracurricular: Participación en actividades extracurriculares
- Sports: Práctica de deportes
- Music: Práctica de música
- Volunteering: Participación en voluntariado
- Attendance: Porcentaje de asistencia
- Sleep: Horas de sueño diarias
- GPA: Promedio de calificaciones (variable objetivo)

## Solución de Problemas

### Error de conexión a MongoDB
- Verificar credenciales
- Verificar firewall de Azure Cosmos DB
- Verificar conectividad a internet

### Error al cargar modelos
- Verificar que todos los archivos .h5 y .pkl estén en el directorio correcto
- Verificar compatibilidad de versiones de TensorFlow y scikit-learn

### Error en visualizaciones
- Verificar que los datos en MongoDB tengan el formato correcto
- Verificar nombres de columnas

## Tecnologías Utilizadas

- Streamlit: Framework web para Python
- TensorFlow/Keras: Red Neuronal
- Scikit-learn: Random Forest y preprocesamiento
- XGBoost: Modelo de gradient boosting
- Plotly: Visualizaciones interactivas
- PyMongo: Conexión con MongoDB
- Pandas/NumPy: Manipulación de datos

## Autores

Proyecto Final - Big Data Analytics
Universidad del Norte

## Licencia

Este proyecto es parte de un trabajo académico.
