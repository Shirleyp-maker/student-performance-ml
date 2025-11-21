import os
import sys

def check_files():
    """Verifica que todos los archivos necesarios estén presentes"""
    
    required_files = {
        'app.py': 'Archivo principal del dashboard',
        'requirements.txt': 'Dependencias del proyecto',
        'student_performance_neural_network.h5': 'Modelo de Red Neuronal',
        'scaler_neural_network.pkl': 'Scaler para normalización',
        'random_forest_model.pkl': 'Modelo Random Forest',
        'xgboost_model.pkl': 'Modelo XGBoost'
    }
    
    optional_files = {
        'model_results.json': 'Resultados de evaluación de modelos'
    }
    
    print("=" * 60)
    print("VERIFICACIÓN DE ARCHIVOS DEL PROYECTO")
    print("=" * 60)
    print()
    
    missing_files = []
    found_files = []
    
    print("Archivos Requeridos:")
    print("-" * 60)
    for file, description in required_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file)
            size_str = f"{size:,} bytes"
            if size > 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.2f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.2f} KB"
            
            print(f"✓ {file:<45} [{size_str}]")
            found_files.append(file)
        else:
            print(f"✗ {file:<45} [FALTANTE]")
            missing_files.append(file)
    
    print()
    print("Archivos Opcionales:")
    print("-" * 60)
    for file, description in optional_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file)
            size_str = f"{size:,} bytes"
            if size > 1024:
                size_str = f"{size / 1024:.2f} KB"
            print(f"✓ {file:<45} [{size_str}]")
        else:
            print(f"○ {file:<45} [OPCIONAL - No encontrado]")
    
    print()
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Archivos encontrados: {len(found_files)}/{len(required_files)}")
    
    if missing_files:
        print(f"Archivos faltantes: {len(missing_files)}")
        print()
        print("ARCHIVOS FALTANTES:")
        for file in missing_files:
            print(f"  - {file}: {required_files[file]}")
        print()
        print("Por favor, asegúrese de tener todos los archivos necesarios")
        print("antes de ejecutar el dashboard.")
        return False
    else:
        print()
        print("✓ Todos los archivos requeridos están presentes")
        print()
        print("Para ejecutar el dashboard:")
        print("  1. Instale las dependencias: pip install -r requirements.txt")
        print("  2. Ejecute el dashboard: streamlit run app.py")
        return True

if __name__ == "__main__":
    success = check_files()
    sys.exit(0 if success else 1)
