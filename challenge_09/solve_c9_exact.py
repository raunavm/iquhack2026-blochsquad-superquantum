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


Sdg = np.array([[1, 0], [0, -1j]])
H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
Z = np.array([[1, 0], [0, -1]])
W_reconst = Sdg @ H @ Z

a = (1+1j)/2
M_target = np.array([
    [1, 0, 0, 0],
    [0, 0, -a, a],
    [0, 1j, 0, 0],
    [0, 0, -a, -a]
], dtype=complex)

qc = QuantumCircuit(2)





qc_temp = QuantumCircuit(1)
W_mat = np.array([[-a, a], [-a, -a]]) * np.array([[1, 0], [0, -1j]]) # This was W = Sdg U_sub.
U_eff_mat = Sdg @ np.array([[-a, a], [-a, -a]])

from qiskit.circuit.library import UnitaryGate
qc_u = QuantumCircuit(1)
qc_u.append(UnitaryGate(U_eff_mat), [0])
qc_u_trans = qiskit.transpile(qc_u, basis_gates=['h', 's', 'sdg', 'x', 'y', 'z', 't', 'tdg']) # Clifford+T basis
print("U_eff decomposition:")
print(qc_u_trans.draw())


qc_full = QuantumCircuit(2)
qc_full.s(0) # Base S

for instr in qc_u_trans.data:
    op_name = instr.operation.name
    if op_name == 'h':
        qc_full.ch(1, 0)
    elif op_name == 's':
        qc_full.t(1)
        qc_full.t(0)
        qc_full.cx(1, 0)
        qc_full.tdg(0)
        qc_full.cx(1, 0)
    elif op_name == 'sdg':
        qc_full.tdg(1)
        qc_full.tdg(0)
        qc_full.cx(1, 0)
        qc_full.t(0)
        qc_full.cx(1, 0)
    elif op_name == 'z':
        qc_full.cz(1, 0)
    elif op_name == 'x':
        qc_full.cx(1, 0)
    elif op_name == 'y':
        qc_full.cy(1, 0) # = S Tdg ...
    else:
        print(f"Unknown op: {op_name}")

qc_full.cx(0, 1)
qc_full.cx(1, 0)
qc_full.cx(0, 1)

final_ond = aligned_ond(M_target, Operator(qc_full).data)
t_count = qc_full.count_ops().get('t', 0) + qc_full.count_ops().get('tdg', 0)
print(f"Final T={t_count}, OND={final_ond}")
qiskit.qasm2.dump(qc_full, "/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_9_exact.qasm")