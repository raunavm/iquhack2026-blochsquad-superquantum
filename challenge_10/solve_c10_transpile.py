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
print(f"KAK ops: {qc_kak.count_ops()}")

basis_gates = ['u3', 'cx'] # First to U3/CX
qc_u3 = transpile(qc_kak, basis_gates=['u3', 'cx'])

target_basis = ['h', 's', 'sdg', 't', 'tdg', 'cx'] # No Rz, Ry

print("Transpiling to Clifford+T with approximation...")

best_qc = None
best_score = (1e9, 1e9) # (dist, t)

for degree in [0.9, 0.99, 0.999]: # Qiskit < 0.23 doesn't support this well, but let's try
    try:
        qc_t = transpile(qc_u3, 
                         basis_gates=target_basis, 
                         optimization_level=3, 
                         approximation_degree=degree,
                         unitary_synthesis_method='sk') 
        
        t_count = qc_t.count_ops().get('t', 0) + qc_t.count_ops().get('tdg', 0)
        
        final_u = Operator(qc_t).data
        def aligned_ond_full(U, V):
            tr = np.trace(U.conj().T @ V)
            return np.sqrt(np.abs(2 - 2 * np.abs(tr) / 2)) # Fast approx metric
            
        dist = aligned_ond_full(U_target, final_u)
        print(f"Degree {degree}: T={t_count}, Dist={dist}")
        
        if dist < best_score[0]: # Prioritize Validity
             best_score = (dist, t_count)
             best_qc = qc_t
             
        if dist < 0.08 and t_count < 634:
            print("Winner found!")
            break
    except Exception as e:
        print(f"Failed degree {degree}: {e}")

if best_qc is None or best_score[0] > 0.08:
    print("Trying default synthesis...")
    try:
        qc_t = transpile(qc_u3, 
                         basis_gates=target_basis, 
                         optimization_level=3, 
                         approximation_degree=0.99)
        t_count = qc_t.count_ops().get('t', 0) + qc_t.count_ops().get('tdg', 0)
        final_u = Operator(qc_t).data
        tr = np.trace(U_target.conj().T @ final_u)
        d = np.sqrt(np.abs(2 - 2 * np.abs(tr) / 2))
        print(f"Default: T={t_count}, Dist={d}")
        if d < best_score[0]: best_qc = qc_t
    except Exception as e:
        print(f"Default failed: {e}")

if best_qc:
    ond_final = 0
    d = U_target
    u = Operator(best_qc).data
    overlap = np.trace(d.conj().T @ u)
    phase = np.angle(overlap)
    ond_final = np.linalg.norm(d - u * np.exp(-1j * phase), ord=2)
    
    t_final = best_qc.count_ops().get('t', 0) + best_qc.count_ops().get('tdg', 0)
    print(f"FINAL: T={t_final}, OND={ond_final}")
    
    qiskit.qasm2.dump(best_qc, "./solution_challenge_10.qasm")
else:
    print("No solution found.")