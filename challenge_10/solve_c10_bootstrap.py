import numpy as np
from qiskit.quantum_info import Operator, random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit.circuit.library import CXGate
from qiskit import QuantumCircuit, transpile
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

def brute_force_base(target_u, trial_count=500000, max_depth=30):
    best_circ = []
    best_dist = 100.0
    best_mat = I
    
    batch_size = 10000
    n_batches = trial_count // batch_size
    T_H = target_u.conj().T
    
    for b in range(n_batches):
        paths = np.random.randint(5, size=(batch_size, max_depth))
        states = np.tile(I, (batch_size, 1, 1))
        
        
        for step in range(max_depth):
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
            
            if best_dist < 0.08: # Good enough for SK base
                return best_circ, best_mat, best_dist
                
    return best_circ, best_mat, best_dist

skd = SolovayKitaev(recursion_degree=2) # Degree 1 might be too weak? 2 is safe.
from qiskit.transpiler import PassManager
pm = PassManager(skd)

def bootstrap_sk(target_u):
    print("    Finding Base (Brute Force)...")
    base_circ, base_mat, base_dist = brute_force_base(target_u, trial_count=1000000, max_depth=35)
    print(f"      Base found: D={base_dist}, T={len([g for g in base_circ if 't' in g])}")
    
    residual = base_mat.conj().T @ target_u
    
    print("    Correcting Residual (SK)...")
    qc_res = QuantumCircuit(1)
    qc_res.unitary(residual, 0)
    
    qc_corr = pm.run(qc_res)
    
    corr_circ = [] # We just append qc_corr to full later
    
    final_u = base_mat @ Operator(qc_corr).data
    
    tr = np.trace(target_u.conj().T @ final_u)
    final_dis = np.sqrt(np.abs(2 - 2 * np.abs(tr) / 2))
    
    return base_circ, qc_corr, final_dis

qc_approx_full = QuantumCircuit(2)
print("Synthesizing Single Qubit Gates (Bootstrap SK)...")

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
        
        base_list, qc_corr, dist = bootstrap_sk(u_target)
        
        for gate_name in base_list:
            if gate_name == 'h': qc_approx_full.h(qubits[0])
            elif gate_name == 's': qc_approx_full.s(qubits[0])
            elif gate_name == 'sdg': qc_approx_full.sdg(qubits[0])
            elif gate_name == 't': qc_approx_full.t(qubits[0])
            elif gate_name == 'tdg': qc_approx_full.tdg(qubits[0])
            
        for sub in qc_corr.data:
             qc_approx_full.append(sub.operation, [qubits[0]])

        t_cnt = qc_approx_full.count_ops().get('t', 0) + qc_approx_full.count_ops().get('tdg', 0)
        print(f"  Cumul Result: T={t_cnt}, Last D={dist}")

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
    qiskit.qasm2.dump(qc_approx_full, "./solution_challenge_10.qasm")