import numpy as np
from qiskit.quantum_info import Operator, random_unitary
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit, transpile
import qiskit.qasm2
from scipy.linalg import block_diag

seed = 42
U_target = random_unitary(4, seed=seed).data
print("Generated Random Unitary (Seed 42).")

from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.circuit.library import CXGate
kak_decomp = TwoQubitBasisDecomposer(gate=CXGate(), euler_basis='U3')
qc_kak = kak_decomp(U_target)

print("KAK Decomposition Circuit:")
print(f"Base Depth: {qc_kak.depth()}, Ops: {qc_kak.count_ops()}")


def approximate_u3(u3_params, recursion_degree=3):
    theta, phi, lam = u3_params
    qc_u3 = QuantumCircuit(1)
    qc_u3.u(theta, phi, lam, 0)
    
    
    basis = ['h', 't', 'tdg', 's', 'sdg', 'x', 'z', 'ry', 'rz'] # No... we want C+T.
    
    
    
    
    try:
        qc_approx = transpile(qc_u3, basis_gates=['h', 's', 'sdg', 't', 'tdg'], optimization_level=3) # , unitary_synthesis_method='sk'
    except Exception as e:
        print(f"Transpile failed: {e}")
        return qc_u3 # Return original if fail
        
    return qc_approx

qc_approx_full = QuantumCircuit(2)

print("Synthesizing Approximation (Degree 2 for speed first)...")
for instr in qc_kak.data:
    op = instr.operation
    try:
        if len(instr.qubits) == 1:
            idx = qc_kak.find_bit(instr.qubits[0]).index
            qubits = [idx]
        else:
            qubits = [qc_kak.find_bit(q).index for q in instr.qubits]
    except Exception:
        qubits = [q._index for q in instr.qubits] # Unsafe fallback

    
    if op.name == 'cx':
        qc_approx_full.cx(qubits[0], qubits[1])
    elif op.name == 'u' or op.name == 'u3': # U3
        qc_app = approximate_u3(op.params, recursion_degree=3)
        for sub_instr in qc_app.data:
            sub_op = sub_instr.operation
            qc_approx_full.append(sub_op, [qubits[0]]) # Single qubit
    elif op.name == 'global_phase':
        pass # Ignore or handle?
    else:
        print(f"Unknown op: {op.name}")

t_count = qc_approx_full.count_ops().get('t', 0) + qc_approx_full.count_ops().get('tdg', 0)

def aligned_ond(U, V):
    d = U
    u = V
    overlap = np.trace(d.conj().T @ u)
    if np.abs(overlap) < 1e-9: phase = 0
    else: phase = np.angle(overlap)
    return np.linalg.norm(d - u * np.exp(-1j * phase), ord=2)

ond = aligned_ond(U_target, Operator(qc_approx_full).data)
print(f"Result: T={t_count}, OND={ond}")

qiskit.qasm2.dump(qc_approx_full, "./solution_challenge_10_approx.qasm")