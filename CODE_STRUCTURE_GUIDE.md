# 機器學習程式碼結構說明指南
# Machine Learning Code Structure Guide

## 📋 程式檔案說明

### 主要程式檔案
- **`mike_SVM_withoutBin.py`**: 原始版本（功能完整但註解較少）
- **`mike_SVM_withoutBin_annotated.py`**: 詳細註解版本（建議學習使用）

## 🏗️ 程式架構說明

### 整體流程 (Overall Flow)
```
資料載入 → 特徵工程 → 資料分割 → 類別平衡 → 
模型訓練 → 性能評估 → 結果比較 → 視覺化 → 分析報告
```

### 詳細步驟解析

#### 第1步：套件導入 (Package Imports)
```python
# 數據處理
import pandas as pd
import numpy as np

# 視覺化
import matplotlib.pyplot as plt
import seaborn as sns

# 機器學習
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
```

**重點概念:**
- `pandas`: 資料框操作的標準工具
- `sklearn`: Python機器學習的核心套件
- `imblearn`: 處理不平衡資料集的專用套件

#### 第2步：資料載入 (Data Loading)
```python
df_drug = pd.read_csv('drug200.csv')
df_drug_original = df_drug.copy()  # 備份原始資料
```

**重點概念:**
- 總是備份原始資料
- 檢查資料大小、欄位、類別分布

#### 第3步：特徵工程 (Feature Engineering)
```python
X = df_drug.drop(["Drug"], axis=1)  # 特徵變數
y = df_drug["Drug"]                 # 目標變數
X = pd.get_dummies(X)               # One-hot編碼
```

**重點概念:**
- **One-hot編碼**: 將類別變數轉為數值變數
  - 例如: 'Sex' → 'Sex_F', 'Sex_M'
  - 機器學習算法只能處理數值資料

#### 第4步：資料分割 (Data Splitting)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)
```

**重點概念:**
- **訓練集**: 用於訓練模型 (70%)
- **測試集**: 用於評估模型性能 (30%)
- **random_state**: 確保結果可重現

#### 第5步：處理類別不平衡 (Handle Class Imbalance)
```python
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
```

**重點概念:**
- **SMOTE**: 合成少數類別過採樣技術
- 平衡各類別樣本數，提高模型公平性
- 只在訓練集使用，測試集保持原始分布

#### 第6步：模型定義 (Model Definition)
```python
classifiers = {
    'Linear SVM': SVC(kernel='linear', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    # ... 更多模型
}
```

**模型選擇說明:**
- **Linear SVM**: 線性分類，解釋性好
- **RBF SVM**: 非線性分類，處理複雜邊界
- **Random Forest**: 集成方法，穩定性佳
- **Gradient Boosting**: 序列集成，準確率高
- **Neural Network**: 深度學習，適合複雜模式

#### 第7步：模型訓練循環 (Model Training Loop)
```python
for name, classifier in classifiers.items():
    # 記錄訓練時間
    start_time = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 進行預測
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 儲存結果
    results[name] = {
        'accuracy': accuracy,
        'train_time': train_time,
        'predictions': y_pred
    }
```

**重點概念:**
- 統一的訓練流程確保公平比較
- 記錄多種性能指標
- 結構化儲存結果便於後續分析

#### 第8步：視覺化 (Visualization)
```python
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子圖1: 準確率比較
axes[0,0].bar(methods, accuracies)

# 子圖2: 訓練時間比較
axes[0,1].bar(methods, train_times)

# 子圖3: 準確率 vs 速度散點圖
axes[1,0].scatter(train_times, accuracies)

# 子圖4: 最佳方法混淆矩陣
axes[1,1].imshow(confusion_matrix)
```

**圖表類型說明:**
- **長條圖**: 比較不同方法的單一指標
- **散點圖**: 展示兩個指標間的關係
- **混淆矩陣**: 詳細分析分類錯誤模式

## 🎯 關鍵機器學習概念

### 1. 訓練 vs 測試
- **訓練集**: 模型學習的資料
- **測試集**: 評估模型泛化能力的資料
- **重要性**: 避免過度配適(overfitting)

### 2. 交叉驗證 (Cross-Validation)
- 更可靠的性能評估方法
- 將資料分成多折進行訓練和驗證
- 減少隨機性影響

### 3. 特徵縮放 (Feature Scaling)
- 某些算法需要特徵在相似尺度上
- 標準化: (x - mean) / std
- 正規化: (x - min) / (max - min)

### 4. 超參數調整 (Hyperparameter Tuning)
- 模型性能的關鍵因素
- 方法: 網格搜索、隨機搜索、貝葉斯優化
- 需要在驗證集上進行

## 📊 評估指標說明

### 1. 準確率 (Accuracy)
```
準確率 = 正確預測數 / 總預測數
```
- 最直觀的指標
- 類別平衡時較為可靠

### 2. 精確率 (Precision)
```
精確率 = 真正例 / (真正例 + 假正例)
```
- 預測為正例中實際為正例的比例

### 3. 召回率 (Recall)
```
召回率 = 真正例 / (真正例 + 假負例)
```
- 實際正例中被正確識別的比例

### 4. F1分數 (F1-Score)
```
F1 = 2 * (精確率 * 召回率) / (精確率 + 召回率)
```
- 精確率和召回率的調和平均

## 🛠️ 程式碼優化建議

### 1. 模組化設計
```python
def load_and_preprocess_data():
    # 資料載入和預處理邏輯
    return X, y

def train_models(X_train, y_train):
    # 模型訓練邏輯
    return results

def visualize_results(results):
    # 視覺化邏輯
    pass
```

### 2. 參數配置
```python
CONFIG = {
    'TEST_SIZE': 0.3,
    'RANDOM_STATE': 42,
    'SMOTE_RANDOM_STATE': 42,
    'MODEL_PARAMS': {
        'SVM': {'kernel': 'linear', 'max_iter': 251},
        'RF': {'n_estimators': 100}
    }
}
```

### 3. 錯誤處理
```python
try:
    results = train_model(X_train, y_train)
except Exception as e:
    print(f"模型訓練失敗: {e}")
    continue
```

## 🔍 除錯技巧

### 1. 資料檢查
```python
print(f"資料形狀: {X.shape}")
print(f"缺失值: {X.isnull().sum()}")
print(f"類別分布: {y.value_counts()}")
```

### 2. 模型診斷
```python
# 檢查模型是否成功訓練
if hasattr(model, 'feature_importances_'):
    print("特徵重要性:", model.feature_importances_)

# 檢查預測合理性
print("預測類別:", np.unique(y_pred))
print("實際類別:", np.unique(y_test))
```

### 3. 性能分析
```python
# 檢查各類別性能
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

## 📚 進階學習方向

### 1. 特徵選擇
- 統計方法: 卡方檢驗、ANOVA
- 嵌入方法: Lasso、Ridge
- 包裝方法: 遞歸特徵消除

### 2. 集成學習
- Bagging: Random Forest, Extra Trees
- Boosting: AdaBoost, Gradient Boosting, XGBoost
- Stacking: 多層模型組合

### 3. 深度學習
- 神經網路架構設計
- 正則化技術
- 優化算法選擇

### 4. 模型解釋性
- SHAP值分析
- LIME局部解釋
- 特徵重要性排序