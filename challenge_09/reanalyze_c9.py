import numpy as np
from qiskit.quantum_info import Operator
from scipy.linalg import block_diag

a = (1+1j)/2
M_target = np.array([
    [1, 0, 0, 0],
    [0, 0, -a, a],
    [0, 1j, 0, 0],
    [0, 0, -a, -a]
], dtype=complex)

print("Target Matrix M:")
print(np.round(M_target, 3))

eigvals = np.linalg.eigvals(M_target)
print("Eigenvalues:", np.round(eigvals, 3))

SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
V = SWAP @ M_target
print("V = SWAP @ M:")
print(np.round(V, 3))


