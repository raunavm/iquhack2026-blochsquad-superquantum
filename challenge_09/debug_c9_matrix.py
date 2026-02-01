import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

a = (1+1j)/2
M_target = np.array([
    [1, 0, 0, 0],
    [0, 0, -a, a],
    [0, 1j, 0, 0],
    [0, 0, -a, -a]
], dtype=complex)

qc = QuantumCircuit(2)

qc.x(1)
qc.t(1)
qc.t(0)
qc.cx(1, 0)
qc.tdg(0)
qc.cx(1, 0)
qc.x(1)

qc.ch(1, 0)

qc.cz(1, 0)

qc.z(1)
qc.t(1)

qc.cx(0, 1) # SWAP
qc.cx(1, 0)
qc.cx(0, 1)

print("Constructed Matrix (with Library CH):")
M_calc = Operator(qc).data
print(np.round(M_calc, 3))

print("Target:")
print(np.round(M_target, 3))

diff = np.linalg.norm(M_calc - M_target)
print(f"Direct Diff: {diff}")
def aligned_ond(U, V):
    overlap = np.trace(U.conj().T @ V)
    if np.abs(overlap) < 1e-9: phase = 0
    else: phase = np.angle(overlap)
    return np.linalg.norm(U - V * np.exp(-1j * phase), ord=2)

print(f"OND (Lib CH): {aligned_ond(M_target, M_calc)}")

qc_ch = QuantumCircuit(2)
qc_ch.s(0)
qc_ch.h(0)
qc_ch.t(0)
qc_ch.cx(1, 0)
qc_ch.tdg(0)
qc_ch.h(0)
qc_ch.sdg(0)

print("Manual CH Matrix:")
print(np.round(Operator(qc_ch).data, 3))
print("Library CH Matrix:")
qc_lib = QuantumCircuit(2)
qc_lib.ch(1, 0)
print(np.round(Operator(qc_lib).data, 3))