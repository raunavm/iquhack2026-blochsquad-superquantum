import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2
from qiskit.quantum_info import Operator

b = (-1+1j)/2
c = (1+1j)/2
d = (-1-1j)/2
M_target = np.array([
    [1, 0, 0, 0],
    [0, 0, b, c],
    [0, 1j, 0, 0],
    [0, 0, b, d]
], dtype=complex)

qc = QuantumCircuit(2)

qc.h(0)

qc.t(1)
qc.t(0)
qc.cx(1, 0) # Control 1 Target 0
qc.tdg(0)
qc.cx(1, 0)

qc.h(0)

qc.s(0)
qc.s(1)

qc.cx(0, 1); qc.cx(1, 0); qc.cx(0, 1)

def aligned_ond(U, V):
    tr = np.trace(U.conj().T @ V)
    return 1 - np.abs(tr) / 4.0

ond = aligned_ond(M_target, Operator(qc).data)
print(f"T=3 Solution OND: {ond}")
t_count = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)
print(f"T Count: {t_count}")

if ond < 1e-6:
    qiskit.qasm2.dump(qc, "./solution_challenge_9.qasm")
