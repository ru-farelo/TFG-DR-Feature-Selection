#!/usr/bin/env python3
"""
🧪 Test para verificar que load_data devuelve tipos correctos
"""

import sys
import os
sys.path.append('code')

def test_load_data_types():
    """Test para verificar tipos de datos"""
    
    print("🧪 === TEST LOAD_DATA TYPES ===\n")
    
    try:
        from code.data_processing import load_data
        import pandas as pd
        import numpy as np
        
        # Verificar qué datasets están disponibles
        data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        if not data_files:
            print("❌ No se encontraron archivos CSV en ./data/")
            return False
        
        # Usar el primer dataset disponible
        dataset_name = data_files[0].replace('.csv', '')
        print(f"📁 Usando dataset: {dataset_name}")
        
        x, y, gene_names = load_data(dataset_name)
        
        print(f"📊 Tipos de datos:")
        print(f"   x: {type(x)} - {x.shape}")
        print(f"   y: {type(y)} - {y.shape if hasattr(y, 'shape') else len(y)}")
        print(f"   gene_names: {type(gene_names)} - {gene_names.shape if hasattr(gene_names, 'shape') else len(gene_names)}")
        
        # Verificar que ambos sean pandas objetos
        x_is_pandas = isinstance(x, pd.DataFrame)
        y_is_pandas = isinstance(y, (pd.Series, pd.DataFrame))
        
        print(f"\n🔍 Verificaciones:")
        print(f"   ✅ x es pandas DataFrame: {x_is_pandas}")
        print(f"   ✅ y es pandas Series/DataFrame: {y_is_pandas}")
        
        # Test de indexación
        if x_is_pandas and y_is_pandas:
            print(f"\n🔍 Test de indexación:")
            try:
                test_indices = [0, 1, 2]
                x_subset = x.iloc[test_indices]
                y_subset = y.iloc[test_indices]
                print(f"   ✅ x.iloc[{test_indices}]: {x_subset.shape}")
                print(f"   ✅ y.iloc[{test_indices}]: {y_subset.shape if hasattr(y_subset, 'shape') else len(y_subset)}")
                return True
            except Exception as e:
                print(f"   ❌ Error en indexación: {e}")
                return False
        else:
            print(f"   ❌ Tipos incorrectos para indexación pandas")
            return False
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🌱 === TEST TIPOS DE DATOS LOAD_DATA ===\n")
    
    # Cambiar al directorio correcto
    if not os.path.exists('data'):
        os.chdir('..')
    
    if not os.path.exists('data'):
        print("❌ No se encuentra el directorio 'data'. Ejecutar desde el directorio raíz del proyecto.")
        sys.exit(1)
    
    success = test_load_data_types()
    
    if success:
        print(f"\n🎯 === RESUMEN ===")
        print(f"✅ load_data devuelve tipos correctos")
        print(f"✅ Indexación pandas funciona")
        print(f"✅ Error de .iloc[] corregido")
        print(f"\n🚀 Ahora el experimento debería funcionar!")
    else:
        print(f"\n❌ Revisar la implementación de load_data")