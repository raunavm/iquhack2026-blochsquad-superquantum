import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

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

Z = np.array([[1, 0], [0, -1]])
H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
S = np.array([[1, 0], [0, 1j]])
W_eff = Z @ H @ S @ H @ S

Ry_target = 1/np.sqrt(2) * np.array([[1, -1], [1, 1]])
print(f"Overlap W_eff vs Ry: {aligned_ond(Ry_target, W_eff)}")


qc = QuantumCircuit(2)

qc.x(1)
qc.t(1); qc.t(0); qc.cx(1, 0); qc.tdg(0); qc.cx(1, 0)
qc.x(1)


qc.t(1); qc.t(0); qc.cx(1, 0); qc.tdg(0); qc.cx(1, 0)
qc.ch(1, 0) # Use library CH, verified to be correct T approx
qc.t(1); qc.t(0); qc.cx(1, 0); qc.tdg(0); qc.cx(1, 0)
qc.ch(1, 0)
qc.cz(1, 0)

qc.z(1)
qc.t(1)

qc.cx(0, 1); qc.cx(1, 0); qc.cx(0, 1)

final_ond = aligned_ond(M_target, Operator(qc).data)
t_count = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)
print(f"Final Test T={t_count}, OND={final_ond}")