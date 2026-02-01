from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CHGate
import numpy as np

qc = QuantumCircuit(2)
qc.ch(0, 1) # Control 0, Target 1

print("Transpiling CH...")
basis = ['h', 's', 'sdg', 't', 'tdg', 'cx']
qc_trans = transpile(qc, basis_gates=basis, optimization_level=3)
print("Depth:", qc_trans.depth())
print("T count:", qc_trans.count_ops().get('t', 0) + qc_trans.count_ops().get('tdg', 0))
print(qc_trans.draw())

from qiskit.quantum_info import Operator
op_lib = Operator(CHGate()).data
op_trans = Operator(qc_trans).data

def aligned_ond(U, V):
    d = U
    u = V
    overlap = np.trace(d.conj().T @ u)
    if np.abs(overlap) < 1e-9: phase = 0
    else: phase = np.angle(overlap)
    return np.linalg.norm(d - u * np.exp(-1j * phase), ord=2)

print("OND:", aligned_ond(op_lib, op_trans))