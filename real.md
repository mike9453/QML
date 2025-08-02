python mike.py
📁 載入藥物分類資料集...
📊 資料集大小：(200, 6)
🏷️ 無順序藥物類別：['DrugY' 'drugC' 'drugX' 'drugA' 'drugB']
📋 類別對應表：
{'DrugY': np.int64(0), 'drugA': np.int64(1), 'drugB': np.int64(2), 'drugC': np.int64(3), 'drugX': np.int64(4)}
🔁 套用 SMOTE 平衡資料分布...
📊 重採樣後資料集大小：(455, 5), 類別分布：{np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)}
📐 標準化特徵...
qiskit*runtime_service.\_resolve_cloud_instances:WARNING:2025-08-03 01:45:45,436: Default instance not set. Searching all available instances.
✅ 可用最空閒的後端為：ibm_torino
🚀 真實量子電腦*量子機器學習訓練...
✅ 量子模型訓練完成！
🔮 真實量子電腦\_執行量子預測...
✅ 量子預測完成

📊 量子機器學習結果報告:
precision recall f1-score support

       DrugY       0.43      0.41      0.42        29
       drugA       0.57      0.63      0.60        27
       drugB       0.74      0.65      0.69        26
       drugC       0.81      0.78      0.79        27
       drugX       0.70      0.75      0.72        28

    accuracy                           0.64       137

macro avg 0.65 0.65 0.65 137
weighted avg 0.64 0.64 0.64 137

Traceback (most recent call last):
File "/home/mike/claude-cli/quantum/QML/mike.py", line 166, in <module>
acc_sim = accuracy_score(y_test, y_pred_sm)
^^^^^^^^^
NameError: name 'y_pred_sm' is not defined. Did you mean: 'y_pred'?
(venv) mike@MSI:~/claude-cli/quantum/QML$
