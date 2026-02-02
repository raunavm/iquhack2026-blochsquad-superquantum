import numpy as np
import qiskit.qasm2
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_unitary, Operator, process_fidelity
from qiskit.transpiler.passes import SolovayKitaev, Optimize1qGatesDecomposition
from qiskit.synthesis import generate_basic_approximations, SolovayKitaevDecomposition
from qiskit.transpiler import PassManager

print("--- SOLVING CHALLENGE 10 (FINAL PARETO) ---")

# 1. Generate Target
target_U = random_unitary(4, seed=42)
qc = QuantumCircuit(2)
qc.unitary(target_U, [0, 1])

# 2. Decompose to KAK
qc_kak = transpile(qc, basis_gates=['u', 'cx'], optimization_level=3)

# 3. Solovay-Kitaev Setup
# STRATEGY: Exclude 'z' from the basis search entirely. 
print("Generating SK basis (no Z, depth 14)...")
basis = generate_basic_approximations(
    basis_gates=['h', 't', 'tdg', 's', 'sdg'], # No Z allowed here!
    depth=14 
)
print(f"Basis size: {len(basis)}")

# Recursion 2: "Goldilocks" zone.
# Use decomposition class directly to avoid PassManager confusion
skd = SolovayKitaevDecomposition(basic_approximations=basis, depth=2)

# 4. Synthesis Loop
final_qc = QuantumCircuit(2)
print("Synthesizing...")

for instruction in qc_kak.data:
    op = instruction.operation
    qubits = instruction.qubits
    indices = [final_qc.find_bit(q).index for q in qubits]
    
    if op.name == 'cx':
        final_qc.cx(indices[0], indices[1])
        
    elif op.name == 'u':
        # SU(2) Normalization Fix
        matrix = op.to_matrix()
        det = np.linalg.det(matrix)
        if abs(det) > 1e-6:
             matrix_su2 = matrix / np.sqrt(det)
        else:
             matrix_su2 = matrix
        
        # SK Synthesis
        # Run directly on matrix
        discretized = skd.run(matrix_su2, recursion_degree=2)
        
        # Manual Append (No Transpile!)
        for gate in discretized.data:
            g_op = gate.operation
            # g_op checks q[0]. Map to indices[0]
            
            # Extra Safety: If Z sneaks in (impossible if basis excluded it, but safe check)
            if g_op.name == 'z':
                final_qc.s(indices[0])
                final_qc.s(indices[0])
            else:
                final_qc.append(g_op, indices)
    
    else:
        final_qc.append(op, indices)

# 5. Optimization 
print("Optimizing...")
pm = PassManager([Optimize1qGatesDecomposition(basis=['h', 't', 'tdg', 's', 'sdg', 'cx'])])
final_qc = pm.run(final_qc)

# 6. Final Validation
final_op = Operator(final_qc)
fid = process_fidelity(target_U, final_op)
ops = final_qc.count_ops()
t_count = ops.get('t', 0) + ops.get('tdg', 0)
approx_dist = np.sqrt(2 * (1 - np.sqrt(fid)))

print(f"\n--- FINAL RESULTS ---")
print(f"T-count: {t_count} (Competitive Range: 400-800)")
print(f"Est. Distance: {approx_dist:.5f} (Target: < 0.01)")
print(f"Has Z gate? {'z' in ops}")

qiskit.qasm2.dump(final_qc, 'challenge_10/solution_challenge_10.qasm')
print("Saved to challenge_10/solution_challenge_10.qasm")
