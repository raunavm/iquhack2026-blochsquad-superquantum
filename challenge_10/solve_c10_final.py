import numpy as np
from qiskit.quantum_info import Operator, random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.circuit.library import CXGate
from qiskit import QuantumCircuit
import qiskit.qasm2
import time

seed = 42
U_target = random_unitary(4, seed=seed).data
kak_decomp = TwoQubitBasisDecomposer(gate=CXGate(), euler_basis='U3')
qc_kak = kak_decomp(U_target)

print(f"KAK ops: {qc_kak.count_ops()}")

S = np.array([[1, 0], [0, 1j]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
I = np.eye(2)

gate_set = [
    ('h', H, 0),
    ('s', S, 0),
    ('sdg', S.conj().T, 0),
    ('t', T, 1),
    ('tdg', T.conj().T, 1)
]

mats_vec = np.array([g[1] for g in gate_set])

def aligned_dist(U, V):
    tr = np.trace(U.conj().T @ V)
    return np.sqrt(np.abs(2 - 2 * np.abs(tr) / 2)) 

def brute_force_search(target_u, trial_count=500000, max_depth=40, target_dist=0.05):
    best_circ = []
    best_dist = 100.0
    best_mat = I
    
    batch_size = 10000
    n_batches = trial_count // batch_size
    if n_batches == 0: n_batches = 1
    
    T_H = target_u.conj().T
    
    for b in range(n_batches):
        for d in [20, 25, 30, 35, 40]:
            paths = np.random.randint(5, size=(batch_size, d))
            
            states = np.tile(I, (batch_size, 1, 1))
            
            for step in range(d):
                chosen = mats_vec[paths[:, step]]
                states = np.matmul(chosen, states)
            
            Product = np.matmul(T_H, states)
            Tr = Product[:, 0, 0] + Product[:, 1, 1]
            Dists = np.sqrt(np.abs(2 - 2 * np.abs(Tr) / 2))
            
            min_idx = np.argmin(Dists)
            min_d = Dists[min_idx]
            
            if min_d < best_dist:
                best_dist = min_d
                best_seq_idx = paths[min_idx]
                best_circ = [gate_set[i][0] for i in best_seq_idx]
                best_mat = states[min_idx]
                
                if best_dist < target_dist:
                    return best_circ, best_mat, best_dist
                    
    return best_circ, best_mat, best_dist

def iterative_brute_force(target_u):
    full_circ = []
    current_u = I
    
    print("    Stage 0 (Base 10M)...")
    seq, mat, dist = brute_force_search(target_u, trial_count=10000000, target_dist=0.02, max_depth=50) # Target 0.02
    full_circ.extend(seq)
    current_u = mat @ current_u
    print(f"      Result: D={dist}, T={len([g for g in full_circ if 't' in g])}")
    
    if dist > 0.01:
        pass
        
    return full_circ, aligned_dist(target_u, current_u)
        
    return full_circ, aligned_dist(target_u, current_u)

qc_approx_full = QuantumCircuit(2)
print("Synthesizing Single Qubit Gates (Iterative Brute Force)...")

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
    
    print(f"Processing Op {op.name} on {qubits}")

    if op.name == 'cx':
        qc_approx_full.cx(qubits[0], qubits[1])
    elif op.name == 'u' or op.name == 'u3':
        theta, phi, lam = op.params
        qc_dummy = QuantumCircuit(1)
        qc_dummy.u(theta, phi, lam, 0)
        u_target = Operator(qc_dummy).data
        
        circ, dist = iterative_brute_force(u_target) 
        
        t_cnt = len([g for g in circ if 't' in g])
        print(f"  Final Gate: T={t_cnt}, D={dist}")

        for gate_name in circ:
            if gate_name == 'h': qc_approx_full.h(qubits[0])
            elif gate_name == 's': qc_approx_full.s(qubits[0])
            elif gate_name == 'sdg': qc_approx_full.sdg(qubits[0])
            elif gate_name == 't': qc_approx_full.t(qubits[0])
            elif gate_name == 'tdg': qc_approx_full.tdg(qubits[0])

def aligned_ond_full(U, V):
    d = U
    u = V
    overlap = np.trace(d.conj().T @ u)
    if np.abs(overlap) < 1e-9: phase = 0
    else: phase = np.angle(overlap)
    return np.linalg.norm(d - u * np.exp(-1j * phase), ord=2)

ond = aligned_ond_full(U_target, Operator(qc_approx_full).data)
t_count = qc_approx_full.count_ops().get('t', 0) + qc_approx_full.count_ops().get('tdg', 0)
print(f"Approximated Solution T={t_count}, OND={ond}")
if ond < 0.1:
    qiskit.qasm2.dump(qc_approx_full, "/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_10.qasm")