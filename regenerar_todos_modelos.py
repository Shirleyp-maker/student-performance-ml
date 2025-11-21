"""
Script para regenerar TODOS los modelos compatibles con el dataset actual de MongoDB
Incluye: Red Neuronal, Scaler, Random Forest y XGBoost
"""

import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import json

print("="*70)
print("REGENERACIÓN COMPLETA DE TODOS LOS MODELOS")
print("="*70)
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
    print(f"   Columnas originales: {len(df.columns)}")
except Exception as e:
    print(f"   ✗ Error cargando datos: {e}")
    exit(1)

# Preparar datos
print("\n3. Preparando datos...")

if 'GPA' not in df.columns:
    print("   ✗ Error: No se encontró la columna 'GPA'")
    exit(1)

# Eliminar columnas no útiles
columns_to_drop = ['Student_ID']
for col in columns_to_drop:
    if col in df.columns:
        df = df.drop(col, axis=1)
        print(f"   Eliminada columna: {col}")

# Identificar columnas categóricas
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'GPA' in categorical_cols:
    categorical_cols.remove('GPA')

print(f"   Columnas categóricas: {len(categorical_cols)}")

# Codificar variables categóricas
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print(f"   ✓ Variables categóricas codificadas")

# Separar características y objetivo
X = df.drop('GPA', axis=1)
y = df['GPA']

print(f"   ✓ Características: {X.shape[1]} columnas")
print(f"   ✓ Objetivo (GPA): {len(y)} valores")

# Dividir datos
print("\n4. Dividiendo datos (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   ✓ Entrenamiento: {len(X_train)} registros")
print(f"   ✓ Prueba: {len(X_test)} registros")

# Crear y guardar Scaler
print("\n5. Creando Scaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open('scaler_neural_network.pkl', 'wb') as f:
    pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

print("   ✓ Scaler guardado")

# Entrenar Random Forest
print("\n6. Entrenando Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

rf_metrics = {
    'train_r2': r2_score(y_train, rf_train_pred),
    'test_r2': r2_score(y_test, rf_test_pred),
    'mae': mean_absolute_error(y_test, rf_test_pred),
    'mse': mean_squared_error(y_test, rf_test_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, rf_test_pred))
}

print(f"   ✓ R² Train: {rf_metrics['train_r2']:.4f}")
print(f"   ✓ R² Test: {rf_metrics['test_r2']:.4f}")
print(f"   ✓ MAE: {rf_metrics['mae']:.4f}")
print(f"   ✓ RMSE: {rf_metrics['rmse']:.4f}")

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f, protocol=pickle.HIGHEST_PROTOCOL)

print("   ✓ Random Forest guardado")

# Entrenar XGBoost
print("\n7. Entrenando XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)

xgb_metrics = {
    'train_r2': r2_score(y_train, xgb_train_pred),
    'test_r2': r2_score(y_test, xgb_test_pred),
    'mae': mean_absolute_error(y_test, xgb_test_pred),
    'mse': mean_squared_error(y_test, xgb_test_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, xgb_test_pred))
}

print(f"   ✓ R² Train: {xgb_metrics['train_r2']:.4f}")
print(f"   ✓ R² Test: {xgb_metrics['test_r2']:.4f}")
print(f"   ✓ MAE: {xgb_metrics['mae']:.4f}")
print(f"   ✓ RMSE: {xgb_metrics['rmse']:.4f}")

with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f, protocol=pickle.HIGHEST_PROTOCOL)

print("   ✓ XGBoost guardado")

# Entrenar Red Neuronal
print("\n8. Entrenando Red Neuronal...")
input_dim = X_train_scaled.shape[1]

nn_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

nn_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print(f"   Arquitectura: {input_dim} → 128 → 64 → 32 → 1")

history = nn_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
    ]
)

nn_train_pred = nn_model.predict(X_train_scaled, verbose=0)
nn_test_pred = nn_model.predict(X_test_scaled, verbose=0)

nn_metrics = {
    'train_r2': r2_score(y_train, nn_train_pred),
    'test_r2': r2_score(y_test, nn_test_pred),
    'mae': mean_absolute_error(y_test, nn_test_pred),
    'mse': mean_squared_error(y_test, nn_test_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, nn_test_pred))
}

print(f"   ✓ R² Train: {nn_metrics['train_r2']:.4f}")
print(f"   ✓ R² Test: {nn_metrics['test_r2']:.4f}")
print(f"   ✓ MAE: {nn_metrics['mae']:.4f}")
print(f"   ✓ RMSE: {nn_metrics['rmse']:.4f}")
print(f"   ✓ Epochs entrenados: {len(history.history['loss'])}")

nn_model.save('student_performance_neural_network.h5')
print("   ✓ Red Neuronal guardada")

# Guardar resultados
print("\n9. Guardando resultados de comparación...")
results = {
    'neural_network': nn_metrics,
    'random_forest': rf_metrics,
    'xgboost': xgb_metrics
}

with open('model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("   ✓ Resultados guardados en model_results.json")

# Verificar archivos
print("\n10. Verificando archivos generados...")
import os

files_to_check = [
    'scaler_neural_network.pkl',
    'random_forest_model.pkl',
    'xgboost_model.pkl',
    'student_performance_neural_network.h5',
    'model_results.json'
]

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

# Resumen de comparación
print("\n" + "="*70)
print("RESUMEN DE MODELOS")
print("="*70)
print(f"\n{'Modelo':<20} {'R² Test':<12} {'MAE':<12} {'RMSE':<12}")
print("-" * 70)
print(f"{'Red Neuronal':<20} {nn_metrics['test_r2']:<12.4f} {nn_metrics['mae']:<12.4f} {nn_metrics['rmse']:<12.4f}")
print(f"{'Random Forest':<20} {rf_metrics['test_r2']:<12.4f} {rf_metrics['mae']:<12.4f} {rf_metrics['rmse']:<12.4f}")
print(f"{'XGBoost':<20} {xgb_metrics['test_r2']:<12.4f} {xgb_metrics['mae']:<12.4f} {xgb_metrics['rmse']:<12.4f}")

# Encontrar el mejor modelo
best_model = max(
    [('Red Neuronal', nn_metrics['test_r2']),
     ('Random Forest', rf_metrics['test_r2']),
     ('XGBoost', xgb_metrics['test_r2'])],
    key=lambda x: x[1]
)

print("\n" + "="*70)
print(f"MEJOR MODELO: {best_model[0]} (R² = {best_model[1]:.4f})")
print("="*70)
print("\nAhora puedes ejecutar el dashboard con:")
print("  streamlit run app.py")
print("\nTodos los modelos están sincronizados con el dataset actual de MongoDB")
