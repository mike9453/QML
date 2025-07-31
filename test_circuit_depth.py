#!/usr/bin/env python3
"""
測試量子電路深度問題的診斷腳本
"""

import numpy as np
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap

print("🔍 診斷量子電路深度問題...")
print("=" * 50)

# 測試參數
feature_dimension = 5
reps = 3

print(f"測試參數: feature_dimension={feature_dimension}, reps={reps}")
print()

# 1. 測試 ZZFeatureMap
print("1️⃣ 測試 ZZFeatureMap:")
try:
    zz_map = ZZFeatureMap(
        feature_dimension=feature_dimension,
        reps=reps,
        entanglement='full'
    )
    print(f"   ✅ ZZFeatureMap 建立成功")
    print(f"   🔹 深度: {zz_map.depth()}")
    print(f"   🔹 參數數量: {zz_map.num_parameters}")
    print(f"   🔹 量子門數量: {len(zz_map.data)}")
    print(f"   🔹 量子比特數: {zz_map.num_qubits}")
    
    # 打印電路結構
    print("   📋 電路結構:")
    print(zz_map.draw(output='text', fold=-1))
    
except Exception as e:
    print(f"   ❌ ZZFeatureMap 失敗: {e}")

print()

# 2. 測試 PauliFeatureMap
print("2️⃣ 測試 PauliFeatureMap:")
try:
    pauli_map = PauliFeatureMap(
        feature_dimension=feature_dimension,
        reps=reps,
        entanglement='full',
        paulis=['Z', 'ZZ']
    )
    print(f"   ✅ PauliFeatureMap 建立成功")
    print(f"   🔹 深度: {pauli_map.depth()}")
    print(f"   🔹 參數數量: {pauli_map.num_parameters}")
    print(f"   🔹 量子門數量: {len(pauli_map.data)}")
    print(f"   🔹 量子比特數: {pauli_map.num_qubits}")
    
    # 打印電路結構
    print("   📋 電路結構:")
    print(pauli_map.draw(output='text', fold=-1))
    
except Exception as e:
    print(f"   ❌ PauliFeatureMap 失敗: {e}")

print()

# 3. 測試不同的 reps 值
print("3️⃣ 測試不同的 reps 值:")
for test_reps in [1, 2, 3, 4, 5]:
    try:
        test_map = ZZFeatureMap(
            feature_dimension=feature_dimension,
            reps=test_reps,
            entanglement='full'
        )
        print(f"   reps={test_reps}: 深度={test_map.depth()}, 參數={test_map.num_parameters}")
    except Exception as e:
        print(f"   reps={test_reps}: 失敗 - {e}")

print()

# 4. 測試不同的 entanglement 模式
print("4️⃣ 測試不同的 entanglement 模式:")
for entangle_mode in ['full', 'linear', 'circular']:
    try:
        test_map = ZZFeatureMap(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entangle_mode
        )
        print(f"   {entangle_mode}: 深度={test_map.depth()}, 參數={test_map.num_parameters}")
    except Exception as e:
        print(f"   {entangle_mode}: 失敗 - {e}")

print()
print("🔍 診斷完成！")