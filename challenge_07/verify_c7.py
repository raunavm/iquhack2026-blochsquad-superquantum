from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_statevector
import qiskit.qasm2
import numpy as np

target_sv = random_statevector(4, seed=42)
qc = qiskit.qasm2.load("./solution_challenge_7_efficient.qasm")
final_sv = Statevector.from_instruction(qc)
fid = np.abs(target_sv.inner(final_sv))**2
t_count = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)
print(f"Efficient Solution (T={t_count}): Fidelity={fid}")