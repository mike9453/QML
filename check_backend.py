#!/usr/bin/env python3
"""檢查 IBM Quantum 後端狀態"""

import warnings
warnings.filterwarnings('ignore')

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    
    # 使用你的 token
    service = QiskitRuntimeService(
        token="JhnudBWd2klu_halJxN_YwsPTQUhED6abIkxrZEF5jdG",
        channel="ibm_cloud"
    )
    
    # 檢查 ibm_torino 後端狀態
    backend = service.backend('ibm_torino')
    
    print("🔍 IBM Quantum 後端狀態檢查")
    print("=" * 50)
    print(f"📡 後端名稱: {backend.name}")
    print(f"🔌 運行狀態: {backend.status().operational}")
    print(f"⚠️  狀態訊息: {backend.status().status_msg}")
    print(f"📊 排隊任務: {backend.status().pending_jobs}")
    print(f"⏱️  平均排隊時間: {getattr(backend.status(), 'estimated_start_time', 'N/A')}")
    
    # 檢查任務歷史
    print("\n📋 最近任務狀態:")
    jobs = service.jobs(limit=5, backend_name='ibm_torino')
    for i, job in enumerate(jobs):
        print(f"  {i+1}. Job ID: {job.job_id()[:8]}... | 狀態: {job.status()} | 時間: {job.creation_date}")
        
except Exception as e:
    print(f"❌ 連接錯誤: {e}")