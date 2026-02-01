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

theta = np.pi / 7
XX = np.kron(np.array([[0,1],[1,0]]), np.array([[0,1],[1,0]]))
YY = np.kron(np.array([[0,-1j],[1j,0]]), np.array([[0,-1j],[1j,0]]))
ZZ = np.kron(np.array([[1,0],[0,-1]]), np.array([[1,0],[0,-1]]))
H1 = XX + YY

eig, vec = np.linalg.eigh(H1)
U_target = vec @ np.diag(np.exp(1j * theta * eig)) @ vec.conj().T

print("Checking Commutator [XX, YY]:")
comm = XX @ YY - YY @ XX
print(f"Norm: {np.linalg.norm(comm)}") # Should be 0

eig_x, vec_x = np.linalg.eigh(XX)
U_XX = vec_x @ np.diag(np.exp(1j * theta * eig_x)) @ vec_x.conj().T

eig_y, vec_y = np.linalg.eigh(YY)
U_YY = vec_y @ np.diag(np.exp(1j * theta * eig_y)) @ vec_y.conj().T

U_decomp = U_XX @ U_YY
print(f"Decomposition Error (should be 0): {aligned_ond(U_target, U_decomp)}")

print(f"Identity OND: {aligned_ond(U_target, np.eye(4))}")


theta_bench = np.pi / 8
eig_z, vec_z = np.linalg.eigh(ZZ)
U_ZZ_bench = vec_z @ np.diag(np.exp(1j * theta_bench * eig_z)) @ vec_z.conj().T

U_XX_bench = U_XX # No
H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])
HH = np.kron(H, H)
U_XX_bench = HH @ U_ZZ_bench @ HH

S = np.array([[1,0],[0,1j]])
Sdg = np.conj(S).T
SS = np.kron(S, S)
SdgSdg = np.kron(Sdg, Sdg)
U_YY_bench = SdgSdg @ U_XX_bench @ SS

U_T2 = U_XX_bench @ U_YY_bench
print(f"T=2 Solution OND: {aligned_ond(U_target, U_T2)}")