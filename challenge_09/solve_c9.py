import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2
from qiskit.quantum_info import Operator

def aligned_ond(U, V):
    d = U
    u = V
    overlap = np.trace(d.conj().T @ u)
    if np.abs(overlap) < 1e-9: phase = 0
    else: phase = np.angle(overlap)
    return np.linalg.norm(d - u * np.exp(-1j * phase), ord=2)

a = (1+1j)/2 # e^i pi/4 / sqrt(2)
M = np.array([
    [1, 0, 0, 0],
    [0, 0, -a, a],
    [0, 1j, 0, 0],
    [0, 0, -a, -a]
], dtype=complex)

   






qc = QuantumCircuit(2)

U_sub = np.array([[-a, a], [-a, -a]])
Sdag = np.array([[1, 0], [0, -1j]])
W = Sdag @ U_sub

from qiskit.synthesis import OneQubitEulerDecomposer
decomp = OneQubitEulerDecomposer(basis='U3')
w_circ = decomp(W)

try:
    from qiskit.quantum_info import Clifford
    c = Clifford(W)
    print("W is Clifford!")
except:
    print("W is NOT Clifford directly.")



print("Solving Challenge 9...")