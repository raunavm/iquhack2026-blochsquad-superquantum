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

S = np.array([[1, 0], [0, 1j]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
I = np.eye(2, dtype=complex)

gate_mats = np.array([H, S, S.conj().T, T, T.conj().T]) 
gate_names = ['h', 's', 'sdg', 't', 'tdg']
gate_costs = np.array([0, 0, 0, 1, 1])

def aligned_dist_vec(Target, Batch):
    T_H = Target.conj().T
    Product = np.matmul(T_H, Batch)
    Tr = Product[:, 0, 0] + Product[:, 1, 1]
    Dist = np.sqrt(np.abs(2 - 2 * np.abs(Tr) / 2))
    return Dist

def beam_search_solve(target_u, max_t=80, beam_width=3000, target_dist=0.012):
    
    current_mats = np.tile(np.eye(2, dtype=complex), (1, 1, 1)) # (1, 2, 2)
    current_costs = np.array([0])
    current_paths = [[]] # List of lists for paths (simpler than array management)
    
    best_sol = (100.0, [])
    
    
    for step in range(100):
        
        n_curr = len(current_mats)
        next_mats = np.matmul(gate_mats[:, None, :, :], current_mats[None, :, :, :])
        next_mats = next_mats.reshape(-1, 2, 2) # (5*W, 2, 2)
        
        next_costs = (gate_costs[:, None] + current_costs[None, :]).reshape(-1)
        
        
        parent_indices = np.tile(np.arange(n_curr), 5)
        gate_indices = np.repeat(np.arange(5), n_curr)
        
        dists = aligned_dist_vec(target_u, next_mats)
        
        min_idx = np.argmin(dists)
        if dists[min_idx] < best_sol[0]:
            pass

        valid_mask = next_costs <= max_t
        
        
        valid_dists = dists[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_dists) == 0:
            break
            
        if len(valid_dists) > beam_width:
            top_k_local = np.argpartition(valid_dists, beam_width)[:beam_width]
            top_indices = valid_indices[top_k_local]
        else:
            top_indices = valid_indices
            
        current_mats = next_mats[top_indices]
        current_costs = next_costs[top_indices]
        
        new_paths = []
        for exact_idx in top_indices:
            p_idx = parent_indices[exact_idx]
            g_idx = gate_indices[exact_idx]
            new_paths.append(current_paths[p_idx] + [gate_names[g_idx]])
        current_paths = new_paths
        
        beam_dists = aligned_dist_vec(target_u, current_mats)
        m_idx = np.argmin(beam_dists)
        m_d = beam_dists[m_idx]
        
        if m_d < best_sol[0]:
            best_sol = (m_d, current_paths[m_idx])
            t_best = len([g for g in best_sol[1] if 't' in g])
            
            if m_d < target_dist:
                print(f"    Target met! D={m_d}, T={t_best}")
                return best_sol[1], best_sol[0]
                
    print(f"    Beam finished. Best D={best_sol[0]}")
    return best_sol[1], best_sol[0]

qc_approx_full = QuantumCircuit(2)
print("Synthesizing Single Qubit Gates (Beam Search)...")

total_t = 0

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
        
        circ, dist = beam_search_solve(u_target, max_t=75, beam_width=3000, target_dist=0.012)
        
        t_cnt = len([g for g in circ if 't' in g])
        print(f"  Result: T={t_cnt}, D={dist}")
        total_t += t_cnt

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

if ond < 0.1 and t_count < 630:
    qiskit.qasm2.dump(qc_approx_full, "/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_10.qasm")
    print("SUCCESS: Valid and Efficient!")
else:
    print("Optimization partial.")