# 真正的 Quantum SVM 設計原理與實現

## 🎯 理論基礎：Quantum SVM 到底是什麼？

### **經典SVM回顧**
```python
# 經典SVM的核心思想
decision_function = Σ αᵢ yᵢ K(xᵢ, x) + b
# 其中 K(xᵢ, x) 是核函數，決定分類性能
```

### **Quantum SVM的核心概念**
```python
# Quantum SVM使用量子核函數
quantum_kernel(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|²
# 其中 φ(x) = U(x)|0⟩ 是量子特徵映射
```

## 📚 學術文獻中的標準做法

### **Havlíček et al. (2019) - Nature論文**
**標準Quantum SVM架構**：

#### **1. ZZFeatureMap確實是標準選擇**
```python
# 學術標準配置
ZZFeatureMap(
    feature_dimension=n,
    reps=2,  # 通常使用2層
    entanglement='full'  # 或 'linear'
)
```

#### **2. 量子特徵映射公式**
```
φ(x) = U_φ(x)|0⟩
其中 U_φ(x) = ∏ᵢ H_i ∏ᵢⱼ e^{i(π-xᵢ)(π-xⱼ)Z_i Z_j} ∏ᵢ e^{i x_i Z_i}
```

#### **3. 量子核函數計算**
```
K(x₁,x₂) = |⟨0|U_φ†(x₁)U_φ(x₂)|0⟩|²
```

## 🔍 我們的問題診斷

### **問題1：ZZFeatureMap被Qiskit優化掉了**

讓我測試真正的ZZFeatureMap行為：

```python
# 測試不同配置的ZZFeatureMap
configs = [
    {'reps': 1, 'entanglement': 'linear'},
    {'reps': 2, 'entanglement': 'linear'},
    {'reps': 1, 'entanglement': 'full'},
    {'reps': 2, 'entanglement': 'full'},
]

for config in configs:
    fm = ZZFeatureMap(feature_dimension=5, **config)
    print(f"Config: {config}")
    print(f"  Depth: {fm.depth()}")
    print(f"  Gates: {len(fm.data)}")
    print(f"  Decomposed depth: {fm.decompose().depth()}")
```

### **問題2：我們可能用錯了參數**

#### **學術論文中的常見配置**：
1. **特徵維度**: 通常 ≤ 4 (因為量子比特限制)
2. **reps**: 1-2 (不是3!)
3. **entanglement**: 'linear' 更常見

## 🧪 正確的Quantum SVM實現

### **方法1：修正ZZFeatureMap參數**
```python
def create_proper_zzfeaturemap(n_features):
    """按照學術標準創建ZZFeatureMap"""
    
    # 如果特徵太多，使用PCA降維到4維以下
    effective_dims = min(n_features, 4)
    
    feature_map = ZZFeatureMap(
        feature_dimension=effective_dims,
        reps=2,  # 學術標準：2層
        entanglement='linear',  # 線性糾纏更穩定
        data_map_func=lambda x: x  # 線性映射
    )
    
    return feature_map
```

### **方法2：使用分解電路**
```python
def create_decomposed_zzfeaturemap(n_features):
    """強制分解ZZFeatureMap獲得真實深度"""
    
    base_map = ZZFeatureMap(
        feature_dimension=n_features,
        reps=2,
        entanglement='linear'
    )
    
    # 關鍵：強制分解
    decomposed_map = base_map.decompose()
    
    print(f"原始深度: {base_map.depth()}")
    print(f"分解深度: {decomposed_map.depth()}")
    
    return decomposed_map
```

### **方法3：實現原始ZZ特徵映射**
```python
def manual_zz_feature_map(n_qubits, x_params):
    """手動實現ZZ特徵映射的數學定義"""
    
    circuit = QuantumCircuit(n_qubits)
    
    # Step 1: Hadamard層 (創建疊加態)
    for i in range(n_qubits):
        circuit.h(i)
    
    # Step 2: ZZ相互作用 (核心的量子特徵編碼)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            # ZZ interaction: e^{i(π-xᵢ)(π-xⱼ)ZᵢZⱼ}
            circuit.cx(i, j)
            circuit.rz(2 * (np.pi - x_params[i]) * (np.pi - x_params[j]), j)
            circuit.cx(i, j)
    
    # Step 3: Z旋轉 (單體特徵編碼)
    for i in range(n_qubits):
        circuit.rz(2 * x_params[i], i)
    
    return circuit
```

## 📊 三種方法的對比測試

讓我實現一個完整的對比測試：

```python
def compare_feature_map_methods():
    """對比三種Quantum SVM特徵映射方法"""
    
    # 載入數據
    X, y, df, le_target = load_and_prepare_quantum_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 特徵標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    methods = [
        {
            'name': '標準ZZFeatureMap',
            'feature_map': ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')
        },
        {
            'name': '分解ZZFeatureMap', 
            'feature_map': ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear').decompose()
        },
        {
            'name': '手動ZZ特徵映射',
            'feature_map': manual_zz_feature_map(4, [Parameter(f'x_{i}') for i in range(4)])
        }
    ]
    
    results = {}
    
    for method in methods:
        print(f"\n🧪 測試 {method['name']}:")
        
        # 創建量子核
        quantum_kernel = FidelityQuantumKernel(feature_map=method['feature_map'])
        
        # 創建QSVM
        qsvm = QSVC(quantum_kernel=quantum_kernel)
        
        # 使用PCA降維到4維 (匹配量子比特數)
        pca = PCA(n_components=4)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # 訓練和預測
        qsvm.fit(X_train_pca, y_train)
        y_pred = qsvm.predict(X_test_pca)
        
        # 計算準確率
        accuracy = accuracy_score(le_target.inverse_transform(y_test), 
                                le_target.inverse_transform(y_pred))
        
        results[method['name']] = {
            'accuracy': accuracy,
            'depth': method['feature_map'].depth(),
            'gates': len(method['feature_map'].data)
        }
        
        print(f"  電路深度: {results[method['name']]['depth']}")
        print(f"  量子門數: {results[method['name']]['gates']}")
        print(f"  準確率: {accuracy:.4f}")
    
    return results
```

## 🎯 正確的Quantum SVM設計原則

### **1. 特徵維度控制**
```python
# 學術標準：4量子比特以下
if n_features > 4:
    # 使用PCA降維
    pca = PCA(n_components=4)
    X_reduced = pca.fit_transform(X)
else:
    X_reduced = X
```

### **2. ZZ特徵映射參數**
```python
# 按照Havlíček論文的標準配置
ZZFeatureMap(
    feature_dimension=4,  # ≤4 量子比特
    reps=2,              # 2層重複
    entanglement='linear' # 線性糾纏 (不是full!)
)
```

### **3. 數據預處理**
```python
# 標準化到 [0, 2π] 範圍
scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
X_scaled = scaler.fit_transform(X)
```

## 💡 關鍵洞察

### **我們之前的錯誤**：
1. **使用了5個量子比特** (應該≤4)
2. **reps=3過多** (標準是2)
3. **entanglement='full'過複雜** (標準是'linear')
4. **沒有正確的數據預處理**

### **正確的做法**：
1. **PCA降維到4維**
2. **使用標準ZZ特徵映射參數**
3. **如果Qiskit優化掉了，強制分解**
4. **正確的特徵縮放範圍**

## 🔬 理論vs實踐

### **理論上**：
- ZZFeatureMap是Quantum SVM的標準選擇
- 有堅實的數學基礎和理論保證

### **實踐中**：
- Qiskit的實現可能有優化問題
- 需要手動干預確保電路按預期工作

## 📋 推薦的實現順序

1. **先測試標準ZZFeatureMap** (4維, reps=2, linear)
2. **如果深度=1，使用分解版本**
3. **如果還不行，手動實現ZZ特徵映射**
4. **最後才考慮完全自定義電路**

你說得對，**ZZFeatureMap確實是正確的選擇**，問題在於我們的參數配置和Qiskit的實現細節！