#!/usr/bin/env python3
"""找到最空閒的量子後端"""

import warnings
warnings.filterwarnings('ignore')

from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(
    token="JhnudBWd2klu_halJxN_YwsPTQUhED6abIkxrZEF5jdG",
    channel="ibm_cloud"
)

print("🔍 查找最空閒的量子後端...")
print("=" * 60)

# 獲取所有可用的真實量子後端
backends = service.backends(simulator=False, operational=True)

backend_info = []
for backend in backends:
    try:
        status = backend.status()
        queue = status.pending_jobs
        backend_info.append({
            'name': backend.name,
            'queue': queue,
            'operational': status.operational,
            'qubits': backend.configuration().n_qubits
        })
    except:
        continue

# 按排隊數排序
backend_info.sort(key=lambda x: x['queue'])

print("📊 所有可用後端 (按排隊數排序):")
for info in backend_info[:10]:  # 顯示前10個最空閒的
    print(f"  📡 {info['name']:<15} | 排隊: {info['queue']:>3} | 量子比特: {info['qubits']:>2}")

print(f"\n✅ 建議使用: {backend_info[0]['name']} (排隊: {backend_info[0]['queue']})")