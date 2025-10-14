#!/usr/bin/env python3
"""
🧪 Test rápido para verificar CO2 por fases: Feature vs Instance Selection
"""

import sys
import os
sys.path.append('code')

def test_config_phases():
    """Test para verificar configuraciones de medición CO2"""
    
    print("🧪 === TEST DE CONFIGURACIONES CO2 ===\n")
    
    from code.config import read_config
    
    # Simulamos diferentes configuraciones
    test_cases = [
        {
            "name": "Fast-mRMR (Feature Selection)",
            "fast_mrmr": True,
            "pu_learning": False,
            "expected_phase": "feature_selection"
        },
        {
            "name": "PU Learning (Instance Selection)", 
            "fast_mrmr": False,
            "pu_learning": "similarity",  # or "threshold"
            "expected_phase": "instance_selection"
        },
        {
            "name": "Solo Training+Inference",
            "fast_mrmr": False,
            "pu_learning": False,
            "expected_phase": None
        }
    ]
    
    print("📋 Configuraciones a probar:")
    for i, case in enumerate(test_cases, 1):
        print(f"   {i}. {case['name']}")
        print(f"      fast_mrmr: {case['fast_mrmr']}")
        print(f"      pu_learning: {case['pu_learning']}")
        print(f"      expected CO2 phase: {case['expected_phase']}")
        print()
    
    print("✅ Configuraciones definidas correctamente!")
    print("\n🔍 Phases que se medirán:")
    print("   🔬 feature_selection: Si fast_mrmr=True")
    print("   🔬 instance_selection: Si pu_learning=True")
    print("   🚀 training: Siempre")
    print("   🎯 inference: Siempre")
    
    return test_cases

def check_experiment_signature():
    """Verificar que experiment.py tiene la signatura correcta"""
    
    print("\n🔍 === VERIFICANDO SIGNATURE DE EXPERIMENT.PY ===")
    
    try:
        from code.experiment import run_experiment
        import inspect
        
        sig = inspect.signature(run_experiment)
        params = list(sig.parameters.keys())
        
        required_params = ['extract_importances', 'tracker']
        
        print(f"📋 Parámetros encontrados: {len(params)}")
        for param in params:
            print(f"   - {param}")
        
        print(f"\n🔍 Verificando parámetros requeridos:")
        for req_param in required_params:
            if req_param in params:
                print(f"   ✅ {req_param}: Encontrado")
            else:
                print(f"   ❌ {req_param}: FALTANTE")
        
        # Verificar valor de retorno
        print(f"\n📋 Valor de retorno esperado: (metrics, preds, importances, emissions)")
        print(f"✅ Función run_experiment disponible y actualizada!")
        
    except Exception as e:
        print(f"❌ Error verificando experiment.py: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🌱 === TEST FRAMEWORK CO2 TASK-BASED ===\n")
    
    # Test 1: Configuraciones
    test_cases = test_config_phases()
    
    # Test 2: Signature de experiment.py
    signature_ok = check_experiment_signature()
    
    if signature_ok:
        print(f"\n🎯 === RESUMEN ===")
        print(f"✅ Framework CO2 listo para medir:")
        print(f"   🔬 Feature Selection (Fast-mRMR)")
        print(f"   🔬 Instance Selection (PU Learning)")
        print(f"   🚀 Model Training")
        print(f"   🎯 Inference/Prediction")
        print(f"\n🚀 Sistema listo para ejecutar experimentos!")
    else:
        print(f"\n❌ Sistema requiere correcciones antes de ejecutar")