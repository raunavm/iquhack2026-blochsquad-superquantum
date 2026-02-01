import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2
from qiskit.quantum_info import Operator


def append_ry_pi4(qc, qubit):
    qc.sdg(qubit)
    qc.h(qubit)
    qc.t(qubit)
    qc.h(qubit)
    qc.s(qubit)

def append_ry_minus_pi4(qc, qubit):
    qc.sdg(qubit)
    qc.h(qubit)
    qc.tdg(qubit)
    qc.h(qubit)
    qc.s(qubit)

def append_cz(qc, c, t):
    qc.h(t)
    qc.cx(c, t)
    qc.h(t)

def append_ch(qc, c, t):
    append_ry_minus_pi4(qc, t)
    qc.cx(c, t)
    append_ry_pi4(qc, t)
    qc.cx(c, t)
    append_cz(qc, c, t)

def append_cs(qc, c, t):
    qc.t(c); qc.t(t); qc.cx(c, t); qc.tdg(t); qc.cx(c, t)

def append_csdg(qc, c, t):
    qc.tdg(c); qc.tdg(t); qc.cx(c, t); qc.t(t); qc.cx(c, t)

qc = QuantumCircuit(2)

qc.h(1); qc.s(1); qc.s(1); qc.h(1)
append_cs(qc, 1, 0)
qc.h(1); qc.s(1); qc.s(1); qc.h(1)

append_csdg(qc, 1, 0)
append_ch(qc, 1, 0)
append_cs(qc, 1, 0)
append_ch(qc, 1, 0)
append_cz(qc, 1, 0) # Was QC.cz in optimized script.

qc.s(1); qc.s(1)
qc.t(1)

qc.cx(0, 1); qc.cx(1, 0); qc.cx(0, 1)

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

ond = aligned_ond(M_target, Operator(qc).data)
t_count = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)
print(f"Final V3 Solution T={t_count}, OND={ond}")
qiskit.qasm2.dump(qc, "./solution_challenge_9.qasm")