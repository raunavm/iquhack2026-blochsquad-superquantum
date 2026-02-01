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

U_target = 0.5 * np.array([
    [1, 1, 1, 1],
    [1, 1j, -1, -1j],
    [1, -1, 1, -1],
    [1, -1j, -1, 1j]
], dtype=complex)

qc_ref = QuantumCircuit(2)
qc_ref.h(1)
qc_ref.cp(np.pi/2, 0, 1) # Control 0, Target 1
qc_ref.h(0)
qc_ref.swap(0, 1) # Standard Little Endian output swap

print("Reference Qiskit QFT Circuit (H1 CP01 H0 SWAP):")

print(f"Ref OND: {aligned_ond(U_target, Operator(qc_ref).data)}")


qc_test = QuantumCircuit(2)
qc_test.h(0)
qc_test.cp(np.pi/2, 1, 0)
qc_test.h(1)
qc_test.swap(0, 1)
print(f"Test (H0 CP10 H1 SWAP) OND: {aligned_ond(U_target, Operator(qc_test).data)}")

qc_cp = QuantumCircuit(2)
qc_cp.cp(np.pi/2, 1, 0)
cp_mat = Operator(qc_cp).data

qc_decomp = QuantumCircuit(2)
qc_decomp.t(1) # T(c)
qc_decomp.t(0) # T(t)
qc_decomp.cx(1, 0)
qc_decomp.tdg(0)
qc_decomp.cx(1, 0)
decomp_mat = Operator(qc_decomp).data

print(f"CP vs Decomp OND: {aligned_ond(cp_mat, decomp_mat)}")

if aligned_ond(cp_mat, decomp_mat) < 1e-9:
    print("Decomposition is correct.")
else:
    print("Decomposition is WRONG.")