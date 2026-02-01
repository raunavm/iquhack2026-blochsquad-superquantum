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

def append_ch_manual(qc, c, t, invert=False):
    if not invert:
        append_ry_pi4(qc, t)
        qc.cx(c, t)
        append_ry_minus_pi4(qc, t)
        qc.cx(c, t)
    else:
        append_ry_minus_pi4(qc, t)
        qc.cx(c, t)
        append_ry_pi4(qc, t)
        qc.cx(c, t)
    
    append_cz(qc, c, t)

def append_cs(qc, c, t):
    qc.t(c); qc.t(t); qc.cx(c, t); qc.tdg(t); qc.cx(c, t)

def append_csdg(qc, c, t):
    qc.tdg(c); qc.tdg(t); qc.cx(c, t); qc.t(t); qc.cx(c, t)

def run_test(invert_ch):
    qc = QuantumCircuit(2)
    qc.h(1); qc.s(1); qc.s(1); qc.h(1)
    append_cs(qc, 1, 0)
    qc.h(1); qc.s(1); qc.s(1); qc.h(1)

    append_csdg(qc, 1, 0)
    append_ch_manual(qc, 1, 0, invert=invert_ch)
    append_cs(qc, 1, 0)
    append_ch_manual(qc, 1, 0, invert=invert_ch)
    append_cs(qc, 1, 0) # CS at end (from optimized file)

    qc.s(1); qc.s(1)
    qc.t(1)

    qc.cx(0, 1); qc.cx(1, 0); qc.cx(0, 1)
    
    return qc

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

qc1 = run_test(False)
ond1 = aligned_ond(M_target, Operator(qc1).data)
t1 = qc1.count_ops().get('t', 0) + qc1.count_ops().get('tdg', 0)
print(f"Normal CH (pi/4 first): T={t1}, OND={ond1}")

qc2 = run_test(True)
ond2 = aligned_ond(M_target, Operator(qc2).data)
print(f"Invert CH (-pi/4 first): OND={ond2}")

if ond1 < 0.4:
    qiskit.qasm2.dump(qc1, "./solution_challenge_9.qasm")
elif ond2 < 0.4:
    qiskit.qasm2.dump(qc2, "./solution_challenge_9.qasm")
else:
    print("Both failed.")