# 匯入套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

from sklearn.svm import SVC


# ========== Step 1: 載入資料 ========== #
df_drug = pd.read_csv('drug200.csv')

# 備份原始 df 
df_drug_original = df_drug.copy()


# One-hot 編碼 + 分類
X = df_drug.drop(["Drug"], axis=1)
y = df_drug["Drug"]
X = pd.get_dummies(X)
X = X.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# SMOTE 過採樣
X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

# ========== Step 2: SVC 訓練 ========== #
SVCclassifier = SVC(kernel='linear', max_iter=251)
SVCclassifier.fit(X_train, y_train)
y_pred_svc = SVCclassifier.predict(X_test)

print("🎯 傳統 SVC 結果：")
print(classification_report(y_test, y_pred_svc))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svc))
print('SVC accuracy is: {:.2f}%'.format(accuracy_score(y_pred_svc, y_test) * 100))

# ========== 機器學習方法比較 ========== #
print("\n" + "="*60)
print("🤖 多種機器學習方法比較")
print("="*60)

# 導入更多機器學習方法
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import time

# Initialize all classifiers with English names
classifiers = {
    'Linear SVM': SVC(kernel='linear', max_iter=251, random_state=42), # 使用線性核的 SVM
    'RBF SVM': SVC(kernel='rbf', max_iter=251, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42), #   使用隨機森林分類器
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42), # 使用決策樹分類器
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42)
}

# 儲存結果
results = {}

print("🚀 開始訓練和評估各種機器學習方法...")
print("-" * 60)

for name, classifier in classifiers.items():
    print(f"正在訓練 {name}...")
    
    # 記錄訓練時間
    start_time = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 預測
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    predict_time = time.time() - start_time
    
    # 計算性能指標
    accuracy = accuracy_score(y_test, y_pred)
    
    # 儲存結果
    results[name] = {
        'accuracy': accuracy,
        'train_time': train_time,
        'predict_time': predict_time,
        'predictions': y_pred
    }
    
    print(f"  ✅ {name} - 準確率: {accuracy:.4f} ({accuracy*100:.2f}%) | 訓練時間: {train_time:.3f}s")

# ========== 結果比較和視覺化 ========== #
print("\n" + "="*60)
print("📊 詳細性能比較")
print("="*60)

# 建立比較 DataFrame
comparison_data = []
for name, result in results.items():
    comparison_data.append({
        '方法': name,
        '準確率 (%)': f"{result['accuracy']*100:.2f}",
        '訓練時間 (s)': f"{result['train_time']:.3f}",
        '預測時間 (s)': f"{result['predict_time']:.3f}"
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('準確率 (%)', ascending=False)

print("🏆 按準確率排序的結果:")
print(comparison_df.to_string(index=False))

# Set English font for matplotlib to avoid encoding issues
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Visualization comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Machine Learning Methods Comprehensive Comparison', fontsize=16, fontweight='bold')

# 1. Accuracy comparison
methods = list(results.keys())
accuracies = [results[method]['accuracy'] for method in methods]

colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
bars1 = axes[0,0].bar(range(len(methods)), accuracies, color=colors)
axes[0,0].set_title('Accuracy Comparison')
axes[0,0].set_ylabel('Accuracy')
axes[0,0].set_xticks(range(len(methods)))
axes[0,0].set_xticklabels(methods, rotation=45, ha='right')
axes[0,0].set_ylim(0, 1)

# Add value labels
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                  f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Training time comparison
train_times = [results[method]['train_time'] for method in methods]
bars2 = axes[0,1].bar(range(len(methods)), train_times, color=colors)
axes[0,1].set_title('Training Time Comparison')
axes[0,1].set_ylabel('Training Time (seconds)')
axes[0,1].set_xticks(range(len(methods)))
axes[0,1].set_xticklabels(methods, rotation=45, ha='right')

# 3. Accuracy vs Speed scatter plot
predict_times = [results[method]['predict_time'] for method in methods]
scatter = axes[1,0].scatter(train_times, accuracies, 
                           c=range(len(methods)), cmap='viridis', s=100, alpha=0.7)
axes[1,0].set_xlabel('Training Time (seconds)')
axes[1,0].set_ylabel('Accuracy')
axes[1,0].set_title('Accuracy vs Training Speed')

# Add method labels
for i, method in enumerate(methods):
    axes[1,0].annotate(method, (train_times[i], accuracies[i]), 
                      xytext=(5, 5), textcoords='offset points', fontsize=8)

# 4. Best method confusion matrix
best_method = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_predictions = results[best_method]['predictions']
cm = confusion_matrix(y_test, best_predictions)

# Get all drug categories
drug_labels = sorted(df_drug['Drug'].unique())
im = axes[1,1].imshow(cm, interpolation='nearest', cmap='Blues')
axes[1,1].set_title(f'Best Method Confusion Matrix\n({best_method})')

# Add text labels
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[1,1].text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

axes[1,1].set_xticks(range(len(drug_labels)))
axes[1,1].set_yticks(range(len(drug_labels)))
axes[1,1].set_xticklabels(drug_labels)
axes[1,1].set_yticklabels(drug_labels)
axes[1,1].set_xlabel('Predicted Labels')
axes[1,1].set_ylabel('True Labels')

plt.tight_layout()
plt.savefig('ml_methods_comparison_english.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 詳細分析報告 ========== #
print(f"\n📋 最佳方法詳細報告 ({best_method}):")
print("-" * 40)
best_accuracy = results[best_method]['accuracy']
print(f"準確率: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"訓練時間: {results[best_method]['train_time']:.3f} 秒")
print(f"預測時間: {results[best_method]['predict_time']:.3f} 秒")
print("\n分類報告:")
print(classification_report(y_test, best_predictions))

# 找出準確率前三名
top3_methods = sorted(results.keys(), key=lambda x: results[x]['accuracy'], reverse=True)[:3]
print(f"\n🏆 準確率前三名:")
for i, method in enumerate(top3_methods, 1):
    acc = results[method]['accuracy']
    print(f"  {i}. {method}: {acc:.4f} ({acc*100:.2f}%)")

# 找出速度最快的方法
fastest_method = min(results.keys(), key=lambda x: results[x]['train_time'])
print(f"\n⚡ 訓練速度最快: {fastest_method} ({results[fastest_method]['train_time']:.3f}s)")

print(f"\n💾 比較圖表已儲存至 ml_methods_comparison.png")
print("🎉 機器學習方法比較完成！")

