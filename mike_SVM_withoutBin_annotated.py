#!/usr/bin/env python3
"""
藥物分類機器學習方法全面比較程式
Drug Classification Machine Learning Methods Comprehensive Comparison

此程式使用多種機器學習方法對藥物分類資料集進行分析，
包括傳統SVM、集成方法、神經網路等，並提供詳細的性能比較和視覺化。

作者: Claude & Mike
日期: 2025-08-03
"""

# ========== 1. 套件導入 (Package Imports) ========== #
import numpy as np                      # 數值計算
import pandas as pd                     # 資料處理
import matplotlib.pyplot as plt         # 視覺化
import seaborn as sns                   # 進階視覺化
import warnings                         # 警告控制
warnings.filterwarnings('ignore')      # 忽略警告訊息，保持輸出清潔

# 機器學習核心套件
from sklearn.model_selection import train_test_split     # 資料分割
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 資料預處理
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # 評估指標
from imblearn.over_sampling import SMOTE               # 處理類別不平衡

# 基礎分類器
from sklearn.svm import SVC                            # 支援向量機

print("📦 所有套件導入完成")
print("🎯 開始藥物分類機器學習分析...")


# ========== 2. 資料載入與初步處理 (Data Loading & Initial Processing) ========== #
print("\n" + "="*60)
print("📁 第一步：資料載入與初步處理")
print("="*60)

# 載入藥物分類資料集
df_drug = pd.read_csv('drug200.csv')
print(f"📊 原始資料集大小：{df_drug.shape}")
print(f"📋 資料集欄位：{list(df_drug.columns)}")
print(f"🏷️  藥物類別：{df_drug['Drug'].unique()}")
print(f"📊 各類別分布：\n{df_drug['Drug'].value_counts()}")

# 備份原始資料，方便後續參考
df_drug_original = df_drug.copy()
print("💾 原始資料已備份")


# ========== 3. 特徵工程 (Feature Engineering) ========== #
print("\n📊 第二步：特徵工程")
print("-" * 40)

# 分離特徵變數(X)和目標變數(y)
X = df_drug.drop(["Drug"], axis=1)      # 移除目標變數，保留所有特徵
y = df_drug["Drug"]                     # 目標變數：藥物類型

print(f"🔹 原始特徵變數 (X)：{X.shape}")
print(f"🔹 原始特徵欄位：{list(X.columns)}")
print(f"🔹 目標變數 (y)：{y.shape}")

# One-hot編碼：將類別變數轉換為數值格式
# 這是機器學習處理類別資料的標準方法
X = pd.get_dummies(X)                   # 自動識別類別欄位並進行編碼
X = X.astype(int)                       # 轉換為整數類型，提高效率

print(f"🔹 One-hot編碼後特徵數：{X.shape[1]}")
print(f"🔹 編碼後特徵欄位：{list(X.columns)}")


# ========== 4. 資料分割 (Data Splitting) ========== #
print("\n📂 第三步：資料分割")
print("-" * 40)

# 分割訓練集和測試集
# test_size=0.3: 30%用於測試，70%用於訓練
# random_state=0: 設定隨機種子，確保結果可重現
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

print(f"🔹 訓練集大小：{X_train.shape}")
print(f"🔹 測試集大小：{X_test.shape}")
print(f"🔹 訓練集類別分布：\n{pd.Series(y_train).value_counts()}")
print(f"🔹 測試集類別分布：\n{pd.Series(y_test).value_counts()}")


# ========== 5. 處理類別不平衡 (Handle Class Imbalance) ========== #
print("\n⚖️  第四步：處理類別不平衡")
print("-" * 40)

# SMOTE (Synthetic Minority Oversampling Technique)
# 用於平衡各類別的樣本數量，提高模型對少數類別的識別能力
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"🔹 SMOTE前訓練集大小：{X_train.shape}")
print(f"🔹 SMOTE後訓練集大小：{X_train_balanced.shape}")
print(f"🔹 SMOTE後類別分布：\n{pd.Series(y_train_balanced).value_counts()}")

# 更新訓練資料為平衡後的版本
X_train, y_train = X_train_balanced, y_train_balanced


# ========== 6. 基礎SVM模型訓練 (Basic SVM Training) ========== #
print("\n" + "="*60)
print("🔥 第五步：基礎SVM模型訓練")
print("="*60)

# 建立線性支援向量機分類器
# kernel='linear': 使用線性核函數，適合線性可分的資料
# max_iter=251: 最大迭代次數，防止訓練時間過長
SVCclassifier = SVC(kernel='linear', max_iter=251)

print("🚀 開始訓練基礎SVM模型...")
SVCclassifier.fit(X_train, y_train)    # 訓練模型
y_pred_svc = SVCclassifier.predict(X_test)    # 進行預測

print("✅ 基礎SVM模型訓練完成")

# 輸出基礎SVM結果
print("\n🎯 基礎SVM分類結果：")
print("📊 詳細分類報告：")
print(classification_report(y_test, y_pred_svc))
print("\n📋 混淆矩陣：")
print(confusion_matrix(y_test, y_pred_svc))
print(f"\n🏆 SVM準確率：{accuracy_score(y_pred_svc, y_test) * 100:.2f}%")


# ========== 7. 多種機器學習方法比較 (Multiple ML Methods Comparison) ========== #
print("\n" + "="*60)
print("🤖 第六步：多種機器學習方法比較")
print("="*60)

# 導入額外的機器學習方法
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import time

print("📦 機器學習方法導入完成")

# 初始化所有分類器字典
# 每個分類器都設定了合適的參數，並使用英文名稱以避免圖表亂碼
classifiers = {
    # 支援向量機家族
    'Linear SVM': SVC(
        kernel='linear',        # 線性核函數
        max_iter=251,          # 最大迭代次數
        random_state=42        # 隨機種子
    ),
    'RBF SVM': SVC(
        kernel='rbf',          # 徑向基函數核
        max_iter=251,          
        random_state=42
    ),
    
    # 集成學習方法
    'Random Forest': RandomForestClassifier(
        n_estimators=100,      # 決策樹數量
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,      # 弱學習器數量
        random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=50,       # 基學習器數量
        random_state=42
    ),
    
    # 線性模型
    'Logistic Regression': LogisticRegression(
        max_iter=1000,         # 最大迭代次數
        random_state=42
    ),
    
    # 鄰近方法
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=5          # 鄰居數量
    ),
    
    # 樹模型
    'Decision Tree': DecisionTreeClassifier(
        random_state=42
    ),
    
    # 機率模型
    'Naive Bayes': GaussianNB(),
    
    # 神經網路
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(100,),  # 隱藏層大小
        max_iter=1000,             # 最大迭代次數
        random_state=42
    )
}

print(f"🔧 已初始化 {len(classifiers)} 種機器學習方法")
for name in classifiers.keys():
    print(f"   📌 {name}")


# ========== 8. 模型訓練與評估循環 (Model Training & Evaluation Loop) ========== #
print("\n🚀 開始訓練和評估各種機器學習方法...")
print("-" * 60)

# 儲存所有結果的字典
results = {}

# 對每種方法進行訓練和評估
for name, classifier in classifiers.items():
    print(f"🔄 正在處理 {name}...")
    
    # 記錄訓練時間
    train_start_time = time.time()
    classifier.fit(X_train, y_train)                    # 訓練模型
    train_time = time.time() - train_start_time
    
    # 記錄預測時間
    predict_start_time = time.time()
    y_pred = classifier.predict(X_test)                 # 進行預測
    predict_time = time.time() - predict_start_time
    
    # 計算準確率
    accuracy = accuracy_score(y_test, y_pred)
    
    # 儲存所有結果到字典
    results[name] = {
        'accuracy': accuracy,           # 準確率
        'train_time': train_time,       # 訓練時間
        'predict_time': predict_time,   # 預測時間
        'predictions': y_pred           # 預測結果
    }
    
    # 輸出當前方法的結果
    print(f"  ✅ {name}:")
    print(f"     🎯 準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"     ⏱️  訓練時間: {train_time:.3f}秒")
    print(f"     ⚡ 預測時間: {predict_time:.3f}秒")

print("🎉 所有機器學習方法訓練完成！")


# ========== 9. 結果統計與分析 (Results Statistics & Analysis) ========== #
print("\n" + "="*60)
print("📊 第七步：詳細性能比較與分析")
print("="*60)

# 建立結果比較表格
comparison_data = []
for name, result in results.items():
    comparison_data.append({
        '方法名稱': name,
        '準確率': f"{result['accuracy']:.4f}",
        '準確率 (%)': f"{result['accuracy']*100:.2f}%",
        '訓練時間 (秒)': f"{result['train_time']:.3f}",
        '預測時間 (秒)': f"{result['predict_time']:.4f}"
    })

# 轉換為DataFrame並按準確率排序
comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('準確率', ascending=False)

print("🏆 所有方法性能排序結果:")
print(comparison_df.to_string(index=False))

# 找出最佳方法
best_method = max(results.keys(), key=lambda x: results[x]['accuracy'])
fastest_method = min(results.keys(), key=lambda x: results[x]['train_time'])
best_accuracy = results[best_method]['accuracy']

print(f"\n🏆 性能分析總結:")
print(f"   🥇 準確率最高: {best_method} ({best_accuracy*100:.2f}%)")
print(f"   ⚡ 訓練最快: {fastest_method} ({results[fastest_method]['train_time']:.3f}秒)")

# 前三名準確率
top3_methods = sorted(results.keys(), key=lambda x: results[x]['accuracy'], reverse=True)[:3]
print(f"\n🏆 準確率前三名:")
for i, method in enumerate(top3_methods, 1):
    acc = results[method]['accuracy']
    print(f"   {i}. {method}: {acc*100:.2f}%")


# ========== 10. 視覺化設定 (Visualization Setup) ========== #
print("\n📊 第八步：建立綜合視覺化分析")
print("-" * 40)

# 設定matplotlib字體，避免中文亂碼
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("🔧 字體設定完成，開始建立圖表...")


# ========== 11. 建立綜合比較圖表 (Create Comprehensive Comparison Charts) ========== #

# 建立2x2子圖格局
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Machine Learning Methods Comprehensive Comparison', 
             fontsize=16, fontweight='bold')

# 準備視覺化資料
methods = list(results.keys())
accuracies = [results[method]['accuracy'] for method in methods]
train_times = [results[method]['train_time'] for method in methods]
predict_times = [results[method]['predict_time'] for method in methods]

# 設定顏色主題
colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))

# === 子圖1: 準確率比較長條圖 ===
bars1 = axes[0,0].bar(range(len(methods)), accuracies, color=colors)
axes[0,0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Accuracy')
axes[0,0].set_xticks(range(len(methods)))
axes[0,0].set_xticklabels(methods, rotation=45, ha='right')
axes[0,0].set_ylim(0, 1.05)

# 在長條圖上添加數值標籤
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                  f'{acc:.3f}', ha='center', va='bottom', 
                  fontsize=9, fontweight='bold')

# === 子圖2: 訓練時間比較長條圖 ===
bars2 = axes[0,1].bar(range(len(methods)), train_times, color=colors)
axes[0,1].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
axes[0,1].set_ylabel('Training Time (seconds)')
axes[0,1].set_xticks(range(len(methods)))
axes[0,1].set_xticklabels(methods, rotation=45, ha='right')

# === 子圖3: 準確率 vs 訓練時間散點圖 ===
scatter = axes[1,0].scatter(train_times, accuracies, 
                           c=range(len(methods)), cmap='viridis', 
                           s=100, alpha=0.7, edgecolors='black')
axes[1,0].set_xlabel('Training Time (seconds)')
axes[1,0].set_ylabel('Accuracy')
axes[1,0].set_title('Accuracy vs Training Speed', fontsize=14, fontweight='bold')

# 添加方法標籤到散點圖
for i, method in enumerate(methods):
    axes[1,0].annotate(method, (train_times[i], accuracies[i]), 
                      xytext=(5, 5), textcoords='offset points', 
                      fontsize=8, alpha=0.8)

# === 子圖4: 最佳方法混淆矩陣 ===
best_predictions = results[best_method]['predictions']
cm = confusion_matrix(y_test, best_predictions)
drug_labels = sorted(df_drug['Drug'].unique())

# 繪製混淆矩陣熱圖
im = axes[1,1].imshow(cm, interpolation='nearest', cmap='Blues')
axes[1,1].set_title(f'Best Method Confusion Matrix\n({best_method})', 
                   fontsize=14, fontweight='bold')

# 在混淆矩陣中添加數值
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[1,1].text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=11, fontweight='bold')

# 設定混淆矩陣的軸標籤
axes[1,1].set_xticks(range(len(drug_labels)))
axes[1,1].set_yticks(range(len(drug_labels)))
axes[1,1].set_xticklabels(drug_labels)
axes[1,1].set_yticklabels(drug_labels)
axes[1,1].set_xlabel('Predicted Labels')
axes[1,1].set_ylabel('True Labels')

# 調整子圖間距並儲存
plt.tight_layout()
plt.savefig('ml_methods_comparison_english.png', dpi=300, bbox_inches='tight')
print("💾 綜合比較圖表已儲存為: ml_methods_comparison_english.png")
plt.show()


# ========== 12. 詳細分析報告 (Detailed Analysis Report) ========== #
print("\n" + "="*60)
print("📋 第九步：詳細分析報告")
print("="*60)

print(f"🏆 最佳方法詳細分析 ({best_method}):")
print("-" * 50)
print(f"🎯 準確率: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"⏱️  訓練時間: {results[best_method]['train_time']:.3f} 秒")
print(f"⚡ 預測時間: {results[best_method]['predict_time']:.4f} 秒")

print(f"\n📊 {best_method} 詳細分類報告:")
print(classification_report(y_test, best_predictions))

print(f"\n🏆 準確率排行榜:")
for i, method in enumerate(top3_methods, 1):
    acc = results[method]['accuracy']
    train_t = results[method]['train_time']
    print(f"  {i}. {method}: {acc:.4f} ({acc*100:.2f}%) | 訓練: {train_t:.3f}s")

print(f"\n⚡ 速度分析:")
print(f"  🥇 訓練最快: {fastest_method} ({results[fastest_method]['train_time']:.3f}s)")

# 找出預測最快的方法
fastest_predict = min(results.keys(), key=lambda x: results[x]['predict_time'])
print(f"  ⚡ 預測最快: {fastest_predict} ({results[fastest_predict]['predict_time']:.4f}s)")


# ========== 13. 總結與建議 (Summary & Recommendations) ========== #
print("\n" + "="*60)
print("🎉 第十步：總結與建議")
print("="*60)

print("📊 實驗總結:")
print(f"  🔹 測試了 {len(classifiers)} 種不同的機器學習方法")
print(f"  🔹 資料集: {df_drug.shape[0]} 個樣本, {X.shape[1]} 個特徵")
print(f"  🔹 最高準確率: {best_accuracy*100:.2f}% ({best_method})")
print(f"  🔹 平均準確率: {np.mean(accuracies)*100:.2f}%")

print(f"\n💡 方法建議:")
if best_accuracy > 0.95:
    print(f"  ✅ {best_method} 表現優異，建議用於生產環境")
else:
    print(f"  ⚠️  所有方法準確率都低於95%，建議進一步調參或特徵工程")

print(f"\n📈 性能特點:")
print(f"  🚀 速度考量: 如需快速訓練，推薦 {fastest_method}")
print(f"  🎯 準確度考量: 如需最高準確率，推薦 {best_method}")

print("\n💾 輸出檔案:")
print("  📊 ml_methods_comparison_english.png - 綜合比較圖表")

print("\n🎉 機器學習方法比較分析完成！")
print("="*60)