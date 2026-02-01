import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2
from qiskit.quantum_info import Operator

S = np.array([[1,0],[0,1j]])
H = 1/np.sqrt(2)*np.array([[1,1],[1,-1]])
Sdg = np.array([[1,0],[0,-1j]])
Ry_reconst = S @ H @ S @ H @ Sdg
Ry_target = 1/np.sqrt(2)*np.array([[1,-1],[1,1]])
print(f"Ry Reconst vs Target Diff: {np.linalg.norm(Ry_reconst - Ry_target)}")

qc = QuantumCircuit(2)

qc.h(1); qc.s(1); qc.s(1); qc.h(1)
qc.t(1); qc.t(0); qc.cx(1, 0); qc.tdg(0); qc.cx(1, 0)
qc.h(1); qc.s(1); qc.s(1); qc.h(1)

qc.tdg(1); qc.tdg(0); qc.cx(1, 0); qc.t(0); qc.cx(1, 0)

qc.ch(1, 0)

qc.t(1); qc.t(0); qc.cx(1, 0); qc.tdg(0); qc.cx(1, 0)

qc.ch(1, 0)

qc.t(1); qc.t(0); qc.cx(1, 0); qc.tdg(0); qc.cx(1, 0)

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
t_count = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)
print(f"Optimized Solution T={t_count}, OND={ond}")
qiskit.qasm2.dump(qc, "./solution_challenge_9.qasm")