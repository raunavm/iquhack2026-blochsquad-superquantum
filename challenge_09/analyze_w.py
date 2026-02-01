import numpy as np
from qiskit.circuit.library import UGate
from qiskit.quantum_info import Operator

a = (1+1j)/2
U_sub = np.array([[-a, a], [-a, -a]])
Sdag = np.array([[1, 0], [0, -1j]])
W = Sdag @ U_sub


print("Analyzing W:")
print(W)

det = W[0,0]*W[1,1] - W[0,1]*W[1,0]
print(f"Det: {det}")

scalar = 1/np.sqrt(det)
W_norm = W * scalar
print("Normalized W (SU2):")
print(W_norm)

tr = np.trace(W_norm)
theta = 2 * np.arccos(np.real(tr)/2) # If imaginary part is small...

print(f"Trace: {tr}")
angle_phase = np.angle(tr)
W_su2 = W_norm * np.exp(-1j * angle_phase)
print("New Trace:", np.trace(W_su2))

from qiskit.synthesis import OneQubitEulerDecomposer
decomp = OneQubitEulerDecomposer(basis='U3')
qc = decomp(W)
print("\nEuler Decomposition:")
print(qc.draw())
for instr in qc.data:
    print(instr.operation.params)