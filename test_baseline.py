#!/usr/bin/env python3
"""
測試baseline性能 - 不使用SMOTE，簡單的量子電路
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC  # 傳統SVM作為對比

# 載入資料
def load_simple_data():
    df = pd.read_csv('drug200.csv')
    
    # 簡單的特徵工程
    le_sex = LabelEncoder()
    le_bp = LabelEncoder()
    le_chol = LabelEncoder()
    le_target = LabelEncoder()
    
    df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
    df['BP_encoded'] = le_bp.fit_transform(df['BP'])
    df['Cholesterol_encoded'] = le_chol.fit_transform(df['Cholesterol'])
    
    # 選擇特徵
    features = ['Age', 'Sex_encoded', 'BP_encoded', 'Cholesterol_encoded', 'Na_to_K']
    X = df[features].values
    y = df['Drug'].values  # 直接使用字串標籤
    
    return X, y, df

print("🔍 Baseline性能測試")
print("=" * 40)

# 載入資料
X, y, df = load_simple_data()

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"訓練集: {len(X_train)} 樣本")
print(f"測試集: {len(X_test)} 樣本")

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n類別分布（訓練集）:")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {label}: {count} 個樣本")

print(f"\n類別分布（測試集）:")
unique, counts = np.unique(y_test, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {label}: {count} 個樣本")

# 1. 測試傳統SVM
print(f"\n1️⃣ 傳統SVM (RBF kernel):")
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
y_pred_rbf = svm_rbf.predict(X_test_scaled)
acc_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"準確率: {acc_rbf:.4f} ({acc_rbf*100:.2f}%)")

print(f"\n2️⃣ 傳統SVM (Linear kernel):")
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train_scaled, y_train)
y_pred_linear = svm_linear.predict(X_test_scaled)
acc_linear = accuracy_score(y_test, y_pred_linear)
print(f"準確率: {acc_linear:.4f} ({acc_linear*100:.2f}%)")

print(f"\n3️⃣ 傳統SVM (Polynomial kernel):")
svm_poly = SVC(kernel='poly', degree=3, random_state=42)
svm_poly.fit(X_train_scaled, y_train)
y_pred_poly = svm_poly.predict(X_test_scaled)
acc_poly = accuracy_score(y_test, y_pred_poly)
print(f"準確率: {acc_poly:.4f} ({acc_poly*100:.2f}%)")

# 顯示最佳結果的詳細報告
best_acc = max(acc_rbf, acc_linear, acc_poly)
if best_acc == acc_rbf:
    best_pred = y_pred_rbf
    best_name = "RBF SVM"
elif best_acc == acc_linear:
    best_pred = y_pred_linear
    best_name = "Linear SVM"
else:
    best_pred = y_pred_poly
    best_name = "Polynomial SVM"

print(f"\n🏆 最佳傳統方法: {best_name} ({best_acc:.4f})")
print(f"\n詳細分類報告:")
print(classification_report(y_test, best_pred))

print(f"\n🎯 總結:")
print(f"  - RBF SVM: {acc_rbf:.4f}")
print(f"  - Linear SVM: {acc_linear:.4f}")
print(f"  - Poly SVM: {acc_poly:.4f}")
print(f"  - 當前QSVM: 0.3500")
print(f"\n❌ QSVM性能遠低於傳統方法，需要修正！")