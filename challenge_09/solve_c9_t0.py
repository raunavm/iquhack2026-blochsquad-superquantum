import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2
from qiskit.quantum_info import Operator


qc = QuantumCircuit(2)
qc.cx(0, 1)
qc.s(0)
qc.cx(1, 0) # Control 1 Target 0
qc.sdg(1)

a = (1+1j)/2
M_target = np.array([
    [1, 0, 0, 0],
    [0, 0, -a, a],
    [0, 1j, 0, 0],
    [0, 0, -a, -a]
], dtype=complex)

def aligned_ond(U, V):
    d = U
    u = V
    overlap = np.trace(d.conj().T @ u)
    if np.abs(overlap) < 1e-9: phase = 0
    else: phase = np.angle(overlap)
    return np.linalg.norm(d - u * np.exp(-1j * phase), ord=2)

ond = aligned_ond(M_target, Operator(qc).data)
print(f"T=0 Solution OND: {ond}")

if ond < 0.1:
    qiskit.qasm2.dump(qc, "./solution_challenge_9.qasm")
    print("Saved T=0 Solution!")
else:
    print("Solution Failed OND check (maybe phase issue?).")