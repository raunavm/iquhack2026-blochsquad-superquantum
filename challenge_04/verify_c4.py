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

theta = np.pi / 7
XX = np.kron(np.array([[0,1],[1,0]]), np.array([[0,1],[1,0]]))
YY = np.kron(np.array([[0,-1j],[1j,0]]), np.array([[0,-1j],[1j,0]]))
H1 = XX + YY

eig, vec = np.linalg.eigh(H1)
U_target = vec @ np.diag(np.exp(1j * theta * eig)) @ vec.conj().T

qc2 = qiskit.qasm2.load("./solution_challenge_4_t2.qasm")
ond2 = aligned_ond(U_target, Operator(qc2).data)
print(f"T=2 Solution OND: {ond2}")

qc_hf = qiskit.qasm2.load("./solution_challenge_4_high_fidelity.qasm")
ond_hf = aligned_ond(U_target, Operator(qc_hf).data)
t_hf = qc_hf.count_ops().get('t', 0) + qc_hf.count_ops().get('tdg', 0)
print(f"High Fidelity (T={t_hf}) OND: {ond_hf}")