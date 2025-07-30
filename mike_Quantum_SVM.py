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

# 備份原始 df 給 TrueQuantumSVM 使用（因為他用的是連續型特徵）
df_drug_original = df_drug.copy()

# Binning the 'Age' column into categories
bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
df_drug['Age_binned'] = pd.cut(df_drug['Age'], bins=bin_age, labels=category_age)
df_drug = df_drug.drop(['Age'], axis=1)

# Binning the 'Na_to_K' column into categories
bin_NatoK = [0, 9, 19, 29, 50]
category_NatoK = ['<10', '10-20', '20-30', '>30']
df_drug['Na_to_K_binned'] = pd.cut(df_drug['Na_to_K'], bins=bin_NatoK, labels=category_NatoK)
df_drug = df_drug.drop(['Na_to_K'], axis=1)

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

# ========== Step 3: 真正的量子 SVM 訓練 ========== #
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# 將 TrueQuantumSVM 類別貼上，或 import 進來
# ...（省略）...

# 使用原始連續型資料來訓練量子模型
def load_for_quantum(df_original):
    le_sex = LabelEncoder()
    le_bp = LabelEncoder()
    le_chol = LabelEncoder()

    df_original['Sex_encoded'] = le_sex.fit_transform(df_original['Sex'])
    df_original['BP_encoded'] = le_bp.fit_transform(df_original['BP'])
    df_original['Cholesterol_encoded'] = le_chol.fit_transform(df_original['Cholesterol'])

    features = ['Age', 'Sex_encoded', 'BP_encoded', 'Cholesterol_encoded']
    X = df_original[features].values
    y = df_original['Drug'].values

    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

Xq_train, Xq_test, yq_train, yq_test = load_for_quantum(df_drug_original)

quantum_svm = TrueQuantumSVM(
    feature_dimension=4,  # 和特徵數相同
    reps=3,
    shots=1024
)

quantum_svm.fit(Xq_train, yq_train)
y_pred_q = quantum_svm.predict(Xq_test)

print("🎯 量子 SVM 結果：")
print(classification_report(yq_test, y_pred_q))
print("Confusion Matrix:\n", confusion_matrix(yq_test, y_pred_q))
print('QSVM accuracy is: {:.2f}%'.format(accuracy_score(yq_test, y_pred_q) * 100))
