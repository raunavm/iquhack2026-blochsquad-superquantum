import numpy as np
from qiskit.quantum_info import Operator
import qiskit.qasm2
from qiskit import QuantumCircuit

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

theta = np.pi / 4
XX = np.kron(np.array([[0,1],[1,0]]), np.array([[0,1],[1,0]]))
YY = np.kron(np.array([[0,-1j],[1j,0]]), np.array([[0,-1j],[1j,0]]))
ZZ = np.kron(np.array([[1,0],[0,-1]]), np.array([[1,0],[0,-1]]))
H2 = XX + YY + ZZ

eig, vec = np.linalg.eigh(H2)
U_target = vec @ np.diag(np.exp(1j * theta * eig)) @ vec.conj().T

print("Checking relation to SWAP:")
SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

ond_swap = aligned_ond(U_target, SWAP)
print(f"SWAP OND: {ond_swap}")

if ond_swap < 1e-9:
    print("Confirmed: Target is equivalent to SWAP (Clifford).")
    
    qc = QuantumCircuit(2)
    qc.swap(0, 1)
    
    qc2 = QuantumCircuit(2)
    qc2.cx(0, 1)
    qc2.cx(1, 0)
    qc2.cx(0, 1)
    
    qiskit.qasm2.dump(qc2, "./solution_challenge_5_swap.qasm")
    t_count = qc2.count_ops().get('t', 0) + qc2.count_ops().get('tdg', 0)
    print(f"Saved T=0 Solution. T-Count: {t_count}")

else:
    print("Target is NOT SWAP.")