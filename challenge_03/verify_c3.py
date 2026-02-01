import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import qiskit.qasm2

def aligned_ond(U, V):
    d = U
    u = V
    overlap = np.trace(d.conj().T @ u)
    if np.abs(overlap) < 1e-9:
        phase = 0
    else:
        phase = np.angle(overlap)
    u_aligned = u * np.exp(-1j * phase)
    return np.linalg.norm(d - u_aligned, ord=2)

target_phi = np.pi / 7
U_target = np.diag([np.exp(1j * target_phi), np.exp(-1j * target_phi), np.exp(-1j * target_phi), np.exp(1j * target_phi)])

qc1 = qiskit.qasm2.load("/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_3_t1.qasm")
ond1 = aligned_ond(U_target, Operator(qc1).data)
print(f"T=1 Solution OND: {ond1}")

qc_sk2 = qiskit.qasm2.load("/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_3_sk_d2.qasm")
ond_sk2 = aligned_ond(U_target, Operator(qc_sk2).data)
t_sk2 = qc_sk2.count_ops().get('t', 0) + qc_sk2.count_ops().get('tdg', 0)
print(f"SK Depth 2 (T={t_sk2}) OND: {ond_sk2}")