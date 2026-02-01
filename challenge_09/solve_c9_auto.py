import numpy as np
from qiskit import QuantumCircuit, transpile
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

print("Transpiling...")
qc_trans = transpile(qc, basis_gates=['h', 's', 'sdg', 't', 'tdg', 'cx'], optimization_level=3)

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

ond = aligned_ond(M_target, Operator(qc_trans).data)
t_count = qc_trans.count_ops().get('t', 0) + qc_trans.count_ops().get('tdg', 0)
print(f"Auto Solution T={t_count}, OND={ond}")
qiskit.qasm2.dump(qc_trans, "/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_9.qasm")