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

qc_flat = transpile(qc_kak, basis_gates=['u3', 'cx'])

basis_gates = ['h', 's', 'sdg', 't', 'tdg', 'cx']
print("Starting Approximation Tuning...")

best_qc = None
best_score = (1e9, 1e9) # (dist, t)

for degree in [0.9, 0.95, 0.99, 0.999, 0.9999]:
    try:
        qc_t = transpile(qc_flat, 
                         basis_gates=basis_gates, 
                         optimization_level=3, 
                         approximation_degree=degree)
        
        t_count = qc_t.count_ops().get('t', 0) + qc_t.count_ops().get('tdg', 0)
        
        final_u = Operator(qc_t).data
        def aligned_ond_full(U, V):
            tr = np.trace(U.conj().T @ V)
            return np.sqrt(np.abs(2 - 2 * np.abs(tr) / 2)) 
            
        dist = aligned_ond_full(U_target, final_u)
        print(f"Degree {degree}: T={t_count}, Dist={dist}")
        
        if dist < 0.08:
            if t_count < best_score[1]: # minimize T *if* valid
                 best_score = (dist, t_count)
                 best_qc = qc_t
             
    except Exception as e:
        print(f"Failed degree {degree}: {e}")

if best_qc:
    ond_final = 0
    d = U_target
    u = Operator(best_qc).data
    overlap = np.trace(d.conj().T @ u)
    phase = np.angle(overlap)
    ond_final = np.linalg.norm(d - u * np.exp(-1j * phase), ord=2)
    
    t_final = best_qc.count_ops().get('t', 0) + best_qc.count_ops().get('tdg', 0)
    print(f"WINNER: T={t_final}, OND={ond_final}")
    
    qiskit.qasm2.dump(best_qc, "./solution_challenge_10.qasm")
else:
    print("No improved approx found. Generating Exact...")
    qc_t = transpile(qc_flat, basis_gates=basis_gates, optimization_level=3)
    qiskit.qasm2.dump(qc_t, "./solution_challenge_10.qasm")