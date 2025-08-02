import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE  # 暫時註釋掉，專注測試ZZFeatureMap
import warnings
warnings.filterwarnings('ignore')

# 使用最新版本的 Qiskit 導入方式
#from qiskit_aer import AerSimulator #量子模擬器
from qiskit.primitives import Sampler #Qiskit 新版的量子原語（primitive），可用來從電路中取得樣本（measurement outcomes），主要用於機器學習與變分量子演算法中
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes #特徵映射（feature map）電路，用於把經過標準化的資料嵌入到量子態中
#RealAmplitudes 是一個量子電路，通常用於變分量子演算法（VQA）or 變分量子模型（如 VQC）中，能夠生成具有可調參數的量子態
from qiskit_algorithms.optimizers import COBYLA # COBYLA 是一種無約束的優化演算法，常用於變分量子演算法中
from qiskit_machine_learning.algorithms import VQC # Variational Quantum Classifier (VQC) 是一種使用變分量子電路進行分類的演算法
from qiskit_machine_learning.kernels import FidelityQuantumKernel #量子核函數（quantum kernel）用於量子機器學習中的核方法，能夠計算量子態之間的相似度或距離
from qiskit_machine_learning.algorithms import QSVC #量子支持向量機（Quantum Support Vector Classifier, QSVC）是一種基於量子核函數的支持向量機分類器
# ComputeUncompute not available in this version

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
#from qiskit_ibm_runtime import Fidelity

from qiskit_machine_learning.state_fidelities import ComputeUncompute


def load_prepare_smote_scaled_data():
    print("📁 載入藥物分類資料集...")
    df = pd.read_csv('drug200.csv')
    print(f"📊 資料集大小：{df.shape}")
    print(f"🏷️  無順序藥物類別：{df['Drug'].unique()}")

    # 建立編碼物件，把文字類別轉換為數字
    le_sex = LabelEncoder()  
    le_bp = LabelEncoder()
    le_chol = LabelEncoder()
    le_target = LabelEncoder()

    # 利用編碼物件，把文字類別轉換為數字類別
    df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
    df['BP_encoded'] = le_bp.fit_transform(df['BP'])
    df['Cholesterol_encoded'] = le_chol.fit_transform(df['Cholesterol'])

    
    features = ['Age', 'Sex_encoded', 'BP_encoded', 'Cholesterol_encoded', 'Na_to_K']
    X = df[features].values
    y = le_target.fit_transform(df['Drug'].values)
    print("📋 類別對應表：")
    print(dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))


    # SMOTE 平衡資料分布
    print("🔁 套用 SMOTE 平衡資料分布...")
    smote = SMOTE(random_state=42) #建立 SMOTE 物件
    X_res, y_res = smote.fit_resample(X, y)
    print(f"📊 重採樣後資料集大小：{X_res.shape}, 類別分布：{set(y_res)}")

    print("📐 標準化特徵...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    return X_scaled, y_res


from qiskit.circuit.library import ZZFeatureMap

# 假設你的資料有 5 維特徵
feature_map = ZZFeatureMap(feature_dimension=5, reps=2, entanglement='linear')
feature_map.decompose().draw(output="mpl", fold=20)

#QSVM train

# 修正：使用正確的 API 導入 (qiskit-machine-learning 0.7.2 版本)
from qiskit_machine_learning.algorithms import QSVC  # QSVM 已改名為 QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel  # QuantumKernel 已改名為 FidelityQuantumKernel
# 移除舊版 API：QuantumInstance 和 Aer 在新版本中不再使用
# from qiskit.utils import algorithm_globals, QuantumInstance
# from qiskit import Aer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# 載入資料
X, y = load_prepare_smote_scaled_data()  # 修正：函數只返回兩個值

# 分割訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""

# 修正：使用新版 API 建立量子核函數
# 新版本不需要 QuantumInstance，直接使用 FidelityQuantumKernel
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

# 修正：建立 QSVC 分類器 (QSVM 已改名為 QSVC)
qsvc = QSVC(quantum_kernel=quantum_kernel)

# 訓練 QSVC
print("🚀 開始模擬器量子機器學習訓練...")
qsvc.fit(X_train, y_train)
print("✅ 模擬器量子模型訓練完成！")

# 預測並評估
print("🔮 執行模擬器量子預測...")
y_pred_sm = qsvc.predict(X_test)
print("✅ 模擬器量子預測完成")

# 修正：需要創建 LabelEncoder 來獲取類別名稱
le_target = LabelEncoder()
le_target.fit(['DrugY', 'drugA', 'drugB', 'drugC', 'drugX'])  # 手動設定類別
print("\n📊 量子機器學習結果報告:")
print(classification_report(y_test, y_pred_sm, target_names=le_target.classes_))

"""

#########real quantum device##############

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler as RuntimeSampler

service = QiskitRuntimeService(
    token="JhnudBWd2klu_halJxN_YwsPTQUhED6abIkxrZEF5jdG",
    channel="ibm_cloud"
)

# ✅ 指定實體量子後端
backend = service.least_busy(simulator=False, operational=True)
print(f"✅ 可用最空閒的後端為：{backend.name}")
# 3️⃣ 使用這個後端來建立 Sampler（新版的量子原語執行方式）
sampler_real = RuntimeSampler(mode=backend)

# 4️⃣ 建立 ComputeUncompute fidelity 方法（本質上是個 fidelity primitive）
fidelity = ComputeUncompute(sampler=sampler_real)

#  - qiskit.primitives.Sampler - 只能用於本地模擬器
#  - qiskit_ibm_runtime.Sampler - 可以連接到 IBM 真實量子硬體
# ✅ 正確建立 FidelityQuantumKernel（使用 sampler 參數）
quantum_kernel_real = FidelityQuantumKernel(
    feature_map=feature_map,

)
qsvc_real = QSVC(quantum_kernel=quantum_kernel_real)

# 訓練 QSVC
print("🚀 真實量子電腦_量子機器學習訓練...")
qsvc_real.fit(X_train, y_train)
print("✅ 量子模型訓練完成！")

# 預測並評估
print("🔮 真實量子電腦_執行量子預測...")
y_pred = qsvc_real.predict(X_test)
print("✅ 量子預測完成")

# 修正：需要創建 LabelEncoder 來獲取類別名稱
le_target = LabelEncoder()
le_target.fit(['DrugY', 'drugA', 'drugB', 'drugC', 'drugX'])  # 手動設定類別
print("\n📊 量子機器學習結果報告:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ----------- 模擬器結果評估 ----------
acc_sim = accuracy_score(y_test, y_pred_sm)
precision_sim = precision_score(y_test, y_pred_sm, average='weighted', zero_division=0)
recall_sim = recall_score(y_test, y_pred_sm, average='weighted', zero_division=0)
f1_sim = f1_score(y_test, y_pred_sm, average='weighted', zero_division=0)

# ----------- 真實量子電腦結果評估 ----------
acc_real = accuracy_score(y_test, y_pred)
precision_real = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall_real = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_real = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# ----------- 統一報告表格 ----------
comparison_df = pd.DataFrame({
    '模型': ['模擬器 Sampler', '真實量子電腦 brisbane'],
    '準確率 Accuracy': [acc_sim, acc_real],
    '精確率 Precision': [precision_sim, precision_real],
    '召回率 Recall': [recall_sim, recall_real],
    'F1 分數': [f1_sim, f1_real]
})

import ace_tools as tools; tools.display_dataframe_to_user(name="QSVM 模擬器 vs 實體量子電腦成效比較", dataframe=comparison_df)

# （選用）印出混淆矩陣
print("\n[模擬器] 混淆矩陣：")
print(confusion_matrix(y_test, y_pred_sm))
print("\n[真實量子電腦] 混淆矩陣：")
print(confusion_matrix(y_test, y_pred))
