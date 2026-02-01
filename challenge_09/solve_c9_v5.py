import numpy as np
from qiskit import QuantumCircuit, transpile
import qiskit.qasm2
from qiskit.quantum_info import Operator


def append_ch_qiskit(qc, c, t):
    qc_ch = QuantumCircuit(2)
    qc_ch.ch(0, 1) # c=0, t=1
    qc_ch_trans = transpile(qc_ch, basis_gates=['h', 's', 'sdg', 't', 'tdg', 'cx'], optimization_level=3)
    
    for instr in qc_ch_trans.data:
        op = instr.operation
        qubits = [c if q == qc_ch_trans.qubits[0] else t for q in instr.qubits]
        qc.append(op, qubits)

def append_cz(qc, c, t):
    qc.h(t)
    qc.cx(c, t)
    qc.h(t)

def append_cs(qc, c, t):
    qc.t(c)
    qc.t(t)
    qc.cx(c, t)
    qc.tdg(t)
    qc.cx(c, t)

def append_csdg(qc, c, t):
    qc.tdg(c)
    qc.tdg(t)
    qc.cx(c, t)
    qc.t(t)
    qc.cx(c, t)

qc = QuantumCircuit(2)

qc.h(1); qc.s(1); qc.s(1); qc.h(1)
append_cs(qc, 1, 0)
qc.h(1); qc.s(1); qc.s(1); qc.h(1)

append_csdg(qc, 1, 0)
append_ch_qiskit(qc, 1, 0)
append_cs(qc, 1, 0)
append_ch_qiskit(qc, 1, 0)
append_cz(qc, 1, 0)

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
print(f"Final V5 Solution T={t_count}, OND={ond}")
qiskit.qasm2.dump(qc, "./solution_challenge_9.qasm")