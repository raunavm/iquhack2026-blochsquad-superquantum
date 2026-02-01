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


qc_cs = QuantumCircuit(2)
qc_cs.t(0)
qc_cs.t(1)
qc_cs.cx(0, 1)
qc_cs.tdg(1)
qc_cs.cx(0, 1)

print("Verifying derived CS decomposition...")
cs_matrix = Operator(qc_cs).data
target_cs = np.diag([1, 1, 1, 1j])
print(f"CS OND: {aligned_ond(target_cs, cs_matrix)}")


qc = QuantumCircuit(2)

qc.h(1)

qc.t(0)
qc.t(1)
qc.cx(0, 1)
qc.tdg(1)
qc.cx(0, 1)

qc.h(0)

qc.cx(0, 1)
qc.cx(1, 0)
qc.cx(0, 1)

U_target = 0.5 * np.array([
    [1, 1, 1, 1],
    [1, 1j, -1, -1j],
    [1, -1, 1, -1],
    [1, -1j, -1, 1j]
], dtype=complex)

final_ond = aligned_ond(U_target, Operator(qc).data)
t_count = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)
print(f"Final Solution T={t_count}, OND={final_ond}")

qiskit.qasm2.dump(qc, "./solution_challenge_8_exact.qasm")