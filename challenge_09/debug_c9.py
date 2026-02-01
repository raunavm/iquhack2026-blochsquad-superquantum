import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import UnitaryGate

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



Sdg = np.array([[1, 0], [0, -1j]])
U_eff_mat = Sdg @ np.array([[-a, a], [-a, -a]])

qc_u = QuantumCircuit(1)
qc_u.y(0)
qc_u.h(0)
qc_u.s(0)
u_circ_mat = Operator(qc_u).data

ratio = U_eff_mat[0,0] / u_circ_mat[0,0]
print(f"Phase Ratio (Target/Circuit): {ratio}")
print(f"Angle: {np.angle(ratio)}")
print(f"Angle / pi: {np.angle(ratio)/np.pi}")



op_test = np.array([[1,0],[0,1j]]) @ 1/np.sqrt(2)*np.array([[1,1],[1,-1]]) @ np.array([[0,-1j],[1j,0]])
print("Checking S H Y vs U_eff_mat...")
