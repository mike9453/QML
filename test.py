from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(
    token="JhnudBWd2klu_halJxN_YwsPTQUhED6abIkxrZEF5jdG",
    channel="ibm_cloud"  # ✅ 必加，不能省略
)

backend = service.least_busy(simulator=False, operational=True)
print(f"✅ 可用最空閒的後端為：{backend.name}")
