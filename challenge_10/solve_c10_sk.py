import numpy as np
from qiskit.quantum_info import Operator, random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer, generate_basic_approximations
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit.circuit.library import CXGate
from qiskit import QuantumCircuit, transpile
import qiskit.qasm2

seed = 42
U_target = random_unitary(4, seed=seed).data
kak_decomp = TwoQubitBasisDecomposer(gate=CXGate(), euler_basis='U3')
qc_kak = kak_decomp(U_target)

print(f"KAK ops: {qc_kak.count_ops()}")

basis_gates = ['h', 't', 'tdg', 's', 'sdg']
print("Generating approximations (depth 2)...")
approx = generate_basic_approximations(basis_gates, depth=2) 

try:
    skd = SolovayKitaev(recursion_degree=1) # Start small
    pass
except:
    pass


qc_approx_full = QuantumCircuit(2)

skd = SolovayKitaev(recursion_degree=3)

print("Synthesizing Gates (SK Degree 3)...")

for instr in qc_kak.data:
    op = instr.operation
    try:
        if len(instr.qubits) == 1:
            idx = qc_kak.find_bit(instr.qubits[0]).index
            qubits = [idx]
        else:
            qubits = [qc_kak.find_bit(q).index for q in instr.qubits]
    except:
        qubits = [q._index for q in instr.qubits]
    
    if op.name == 'cx':
        qc_approx_full.cx(qubits[0], qubits[1])
    elif op.name == 'u' or op.name == 'u3':
        qc_u = QuantumCircuit(1)
        qc_u.append(op, [0])
        
        from qiskit.transpiler import PassManager
        pm = PassManager(skd)
        u_gate = op.to_matrix()
        qc_u_unitary = QuantumCircuit(1)
        qc_u_unitary.unitary(u_gate, 0)
        
        qc_synth = pm.run(qc_u_unitary)
        
        for sub_instr in qc_synth.data:
            qc_approx_full.append(sub_instr.operation, [qubits[0]]) # Map to qubit

def aligned_ond(U, V):
    tr = np.trace(U.conj().T @ V)
    return np.sqrt(np.abs(2 - 2 * np.abs(tr) / 2)) # Approx Metric

def aligned_ond_full(U, V):
    d = U
    u = V
    overlap = np.trace(d.conj().T @ u)
    if np.abs(overlap) < 1e-9: phase = 0
    else: phase = np.angle(overlap)
    return np.linalg.norm(d - u * np.exp(-1j * phase), ord=2)

final_u = Operator(qc_approx_full).data
ond = aligned_ond_full(U_target, final_u)
t_count = qc_approx_full.count_ops().get('t', 0) + qc_approx_full.count_ops().get('tdg', 0)
print(f"SK Solution T={t_count}, OND={ond}")

if ond < 0.1:
    qiskit.qasm2.dump(qc_approx_full, "/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_10.qasm")