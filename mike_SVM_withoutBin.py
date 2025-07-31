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
from quantum_svm_fixed import TrueQuantumSVM


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

