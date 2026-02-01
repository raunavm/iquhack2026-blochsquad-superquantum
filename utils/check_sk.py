try:
    from qiskit.synthesis import SolovayKitaev
    print("Found qiskit.synthesis.SolovayKitaev")
except ImportError:
    print("Not found in qiskit.synthesis")

try:
    from qiskit.transpiler.passes.synthesis import SolovayKitaev
    print("Found qiskit.transpiler.passes.synthesis.SolovayKitaev")
except ImportError:
    print("Not found in transpiler passes")

import qiskit
print(f"Qiskit version: {qiskit.__version__}")