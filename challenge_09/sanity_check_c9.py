import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2
from qiskit.quantum_info import Operator

qc = QuantumCircuit(2)

qc.h(1); qc.s(1); qc.s(1); qc.h(1)
qc.t(1); qc.t(0); qc.cx(1, 0); qc.tdg(0); qc.cx(1, 0)
qc.h(1); qc.s(1); qc.s(1); qc.h(1)

qc.tdg(1); qc.tdg(0); qc.cx(1, 0); qc.t(0); qc.cx(1, 0)
qc.ch(1, 0)
qc.t(1); qc.t(0); qc.cx(1, 0); qc.tdg(0); qc.cx(1, 0)
qc.ch(1, 0)
qc.cz(1, 0)

qc.s(1); qc.s(1)
qc.t(1)

qc.cx(0, 1); qc.cx(1, 0); qc.cx(0, 1)

def aligned_ond(U, V):
    d = U
    u = V
    overlap = np.trace(d.conj().T @ u)
    if np.abs(overlap) < 1e-9: phase = 0
    else: phase = np.angle(overlap)
    return np.linalg.norm(d - u * np.exp(-1j * phase), ord=2)

a = (1+1j)/2
M_target = np.array([
    [1, 0, 0, 0],
    [0, 0, -a, a],
    [0, 1j, 0, 0],
    [0, 0, -a, -a]
], dtype=complex)

ond = aligned_ond(M_target, Operator(qc).data)
print(f"Sanity Check (Optimized Logic) OND={ond}")