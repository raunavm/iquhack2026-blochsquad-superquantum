import numpy as np
from qiskit.quantum_info import Operator, random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.circuit.library import CXGate
from qiskit import QuantumCircuit, transpile
import qiskit.qasm2

seed = 42
U_target = random_unitary(4, seed=seed).data
print("Generated U(Seed 42).")

kak_decomp = TwoQubitBasisDecomposer(gate=CXGate(), euler_basis='U3')
qc_kak = kak_decomp(U_target)

basis_gates = ['h', 's', 'sdg', 't', 'tdg', 'cx']

print("Transpiling (Default Validity)...")
qc_t = transpile(qc_kak, basis_gates=basis_gates, optimization_level=3)

def aligned_ond_full(U, V):
    d = U
    u = V
    overlap = np.trace(d.conj().T @ u)
    if np.abs(overlap) < 1e-9: phase = 0
    else: phase = np.angle(overlap)
    return np.linalg.norm(d - u * np.exp(-1j * phase), ord=2)

ond = aligned_ond_full(U_target, Operator(qc_t).data)
t_count = qc_t.count_ops().get('t', 0) + qc_t.count_ops().get('tdg', 0)

print(f"Fallback Solution: T={t_count}, OND={ond}")

if ond < 0.1:
    print("VALID. Saving.")
    qiskit.qasm2.dump(qc_t, "/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_10.qasm")
else:
    print("Warning: Fallback invalid?")