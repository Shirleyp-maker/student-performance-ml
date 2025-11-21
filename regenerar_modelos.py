"""
Script para regenerar scaler y Random Forest compatibles con Python 3.12
Este script se conecta a MongoDB, carga los datos y regenera los modelos
"""

import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

print("="*60)
print("REGENERACIÓN DE MODELOS COMPATIBLES")
print("="*60)
print()

# Conectar a MongoDB
print("1. Conectando a MongoDB Azure Cosmos DB...")
connection_string = "mongodb+srv://shirleyp:Bigdata2$@student-performance-mongo.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false"

try:
    client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
    client.server_info()
    print("   ✓ Conexión exitosa")
except Exception as e:
    print(f"   ✗ Error de conexión: {e}")
    exit(1)

# Cargar datos
print("\n2. Cargando datos desde MongoDB...")
try:
    db = client['student_performance']
    collection = db['students']
    data = list(collection.find())
    df = pd.DataFrame(data)
    
    if '_id' in df.columns:
        df = df.drop('_id', axis=1)
    
    print(f"   ✓ Datos cargados: {len(df)} registros")
    print(f"   Columnas: {list(df.columns)}")
except Exception as e:
    print(f"   ✗ Error cargando datos: {e}")
    exit(1)

# Preparar datos
print("\n3. Preparando datos...")

# Verificar que GPA existe
if 'GPA' not in df.columns:
    print("   ✗ Error: No se encontró la columna 'GPA'")
    exit(1)

# Eliminar columnas no útiles para predicción
columns_to_drop = ['Student_ID', '_id']  # IDs no son útiles
for col in columns_to_drop:
    if col in df.columns:
        df = df.drop(col, axis=1)
        print(f"   Eliminada columna: {col}")

# Identificar columnas categóricas y numéricas
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'GPA' in categorical_cols:
    categorical_cols.remove('GPA')

print(f"   Columnas categóricas encontradas: {len(categorical_cols)}")

# Codificar variables categóricas
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"   Codificada columna: {col}")

# Separar características y objetivo
X = df.drop('GPA', axis=1)
y = df['GPA']

print(f"   ✓ Características: {X.shape[1]} columnas")
print(f"   ✓ Objetivo (GPA): {len(y)} valores")
print(f"   Columnas finales: {list(X.columns)}")

# Dividir datos
print("\n4. Dividiendo datos...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   ✓ Entrenamiento: {len(X_train)} registros")
print(f"   ✓ Prueba: {len(X_test)} registros")

# Crear y guardar Scaler
print("\n5. Creando y guardando Scaler...")
try:
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    with open('scaler_neural_network.pkl', 'wb') as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("   ✓ Scaler guardado exitosamente")
except Exception as e:
    print(f"   ✗ Error guardando scaler: {e}")

# Crear y guardar Random Forest
print("\n6. Entrenando Random Forest...")
try:
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluar
    train_score = rf_model.score(X_train, y_train)
    test_score = rf_model.score(X_test, y_test)
    
    print(f"   ✓ Modelo entrenado")
    print(f"   R² Train: {train_score:.4f}")
    print(f"   R² Test: {test_score:.4f}")
    
    # Guardar
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("   ✓ Random Forest guardado exitosamente")
except Exception as e:
    print(f"   ✗ Error con Random Forest: {e}")

# Verificar archivos
print("\n7. Verificando archivos generados...")
import os

files_to_check = ['scaler_neural_network.pkl', 'random_forest_model.pkl']
for file in files_to_check:
    if os.path.exists(file):
        size = os.path.getsize(file)
        if size > 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.2f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.2f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"   ✓ {file}: {size_str}")
    else:
        print(f"   ✗ {file}: No encontrado")

print("\n" + "="*60)
print("PROCESO COMPLETADO")
print("="*60)
print("\nAhora puedes ejecutar el dashboard con:")
print("  streamlit run app.py")
