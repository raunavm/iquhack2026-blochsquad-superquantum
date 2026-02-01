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


qc = QuantumCircuit(2)

qc.x(1)
qc.t(1)
qc.t(0)
qc.cx(1, 0)
qc.tdg(0)
qc.cx(1, 0)
qc.x(1)

qc.s(0) # on target
qc.h(0)
qc.t(0)
qc.cx(1, 0)
qc.tdg(0)
qc.h(0)
qc.sdg(0)

qc.cz(1, 0)

qc.z(1)
qc.t(1)

qc.cx(0, 1)
qc.cx(1, 0)
qc.cx(0, 1)

a = (1+1j)/2
M_target = np.array([
    [1, 0, 0, 0],
    [0, 0, -a, a],
    [0, 1j, 0, 0],
    [0, 0, -a, -a]
], dtype=complex)

final_ond = aligned_ond(M_target, Operator(qc).data)
t_count = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)
print(f"Final Solution T={t_count}, OND={final_ond}")
qiskit.qasm2.dump(qc, "/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_9.qasm")