from qiskit import QuantumCircuit, transpile
import qiskit.qasm2
import numpy as np
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutativeCancellation, Optimize1qGatesDecomposition

qc = qiskit.qasm2.load("/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_7_sk_d1.qasm")
print(f"Original T Count: {qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)}")

pm = PassManager([CommutativeCancellation(), Optimize1qGatesDecomposition()])
qc_opt = pm.run(qc)

qc_trans = transpile(qc_opt, basis_gates=['h', 's', 'sdg', 't', 'tdg', 'cx'], optimization_level=3)

t_count = qc_trans.count_ops().get('t', 0) + qc_trans.count_ops().get('tdg', 0)
print(f"Qiskit Opt T Count: {t_count}")

qiskit.qasm2.dump(qc_trans, "/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_7_opt.qasm")



targets = [
    (np.pi/2, 0.27221, 4.3799),
    (2.2215, 4.0137, 1.5114),
    (0.90157, 1.7606, 0)
]

print("Searching for better approximations for exact unitaries...")
from qiskit.circuit.library import UGate
from qiskit.quantum_info import Operator

def aligned_ond(U, V):
    d = U
    u = V
    overlap = np.trace(d.conj().T @ u)
    if np.abs(overlap) < 1e-9: phase = 0
    else: phase = np.angle(overlap)
    return np.linalg.norm(d - u * np.exp(-1j * phase), ord=2)

total_t = 0
for i, (theta, phi, lam) in enumerate(targets):
    target_u = UGate(theta, phi, lam).to_matrix()
    
    
    best_ond_local = 1.0
    best_seq = []
    
    import random
    ops = {
        'h': 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex),
        's': np.array([[1, 0], [0, 1j]], dtype=complex),
        't': np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex),
    }
    ops['sdg'] = np.conj(ops['s']).T
    ops['tdg'] = np.conj(ops['t']).T
    op_keys = list(ops.keys())

    for _ in range(50000):
        u = np.eye(2, dtype=complex)
        seq = []
        for _ in range(random.randint(10, 40)): # Length ~30 -> T ~ 10-15?
             k = random.choice(op_keys)
             u = ops[k] @ u
             seq.append(k)
        
        ond = aligned_ond(target_u, u)
        if ond < best_ond_local:
            best_ond_local = ond
            best_seq = seq
            
    t_local = best_seq.count('t') + best_seq.count('tdg')
    print(f"Unitary {i}: T={t_local}, OND={best_ond_local:.5f}")
    total_t += t_local

print(f"Est Total T: {total_t}")