#!/bin/bash

echo "🚀 安裝真正的量子機器學習環境"
echo "=================================="

# 升級 pip
echo "📦 升級 pip..."
pip install --upgrade pip

# 安裝最新版 Qiskit 套件
echo "⚛️  安裝 Qiskit 核心套件..."
pip install qiskit>=1.0.0

echo "🔧 安裝 Qiskit Aer 模擬器..."
pip install qiskit-aer>=0.14.0

echo "🧠 安裝 Qiskit 機器學習套件..."
pip install qiskit-machine-learning>=0.8.0

echo "📊 安裝 Qiskit 演算法套件..."
pip install qiskit-algorithms>=0.3.0

# 安裝基礎科學計算套件
echo "📈 安裝科學計算套件..."
pip install numpy pandas matplotlib seaborn scikit-learn

echo "✅ 安裝完成！"
echo ""
echo "🧪 測試量子環境..."
python -c "
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    print('✅ 所有量子套件導入成功！')
    print('⚛️  量子環境準備就緒')
except ImportError as e:
    print(f'❌ 導入失敗: {e}')
"

echo ""
echo "🎯 現在可以執行真正的量子SVM："
echo "python true_quantum_svm.py"