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

def super_beam_search(target_u, max_t=80, beam_width=10000, target_dist=0.02):
    current_mats = np.tile(I, (1, 1, 1))
    current_costs = np.array([0])
    current_paths = [[]]
    
    best_sol = (100.0, [])
    
    for step in range(max_t + 10): # Depth limit a bit higher than T max
        n_curr = len(current_mats)
        next_mats = np.matmul(gate_mats[:, None, :, :], current_mats[None, :, :, :]).reshape(-1, 2, 2)
        next_costs = (gate_costs[:, None] + current_costs[None, :]).reshape(-1)
        
        parent_indices = np.tile(np.arange(n_curr), 5)
        gate_indices = np.repeat(np.arange(5), n_curr)
        
        dists = aligned_dist_vec(target_u, next_mats)
        
        valid_mask = next_costs <= max_t
        if not np.any(valid_mask): break
        
        valid_dists = dists[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        
        if len(valid_dists) > beam_width:
             top_k_idx = np.argpartition(valid_dists, beam_width)[:beam_width]
             top_indices = valid_indices[top_k_idx]
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
        
        m_idx = np.argmin(valid_dists[top_k_idx if len(valid_dists) > beam_width else range(len(valid_dists))])
        m_d = valid_dists[top_k_idx if len(valid_dists) > beam_width else range(len(valid_dists))][m_idx]
        
        if m_d < best_sol[0]:
            best_sol = (m_d, current_paths[m_idx])
            t_curr = len([g for g in best_sol[1] if 't' in g])
            
            if m_d < target_dist:
                return best_sol[1], best_sol[0]
                
    return best_sol[1], best_sol[0]

qc_full = QuantumCircuit(2)
print("Synthesizing (Super Beam)...")

total_t = 0
total_ond_sq = 0

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
        qc_full.cx(qubits[0], qubits[1])
    elif op.name == 'u' or op.name == 'u3':
        theta, phi, lam = op.params
        qc_dummy = QuantumCircuit(1)
        qc_dummy.u(theta, phi, lam, 0)
        u_target = Operator(qc_dummy).data
        
        circ, dist = super_beam_search(u_target, beam_width=10000, target_dist=0.03) # target 0.03 per gate
        
        t_cnt = len([g for g in circ if 't' in g])
        print(f"  Result: T={t_cnt}, D={dist}")
        total_t += t_cnt
        total_ond_sq += dist**2

        for gate_name in circ:
            if gate_name == 'h': qc_full.h(qubits[0])
            elif gate_name == 's': qc_full.s(qubits[0])
            elif gate_name == 'sdg': qc_full.sdg(qubits[0])
            elif gate_name == 't': qc_full.t(qubits[0])
            elif gate_name == 'tdg': qc_full.tdg(qubits[0])

ond_final = aligned_dist_vec(U_target, Operator(qc_full).data.reshape(1, 4, 4))[0]
def aligned_ond_full(U, V):
    d = U
    u = V
    overlap = np.trace(d.conj().T @ u)
    if np.abs(overlap) < 1e-9: phase = 0
    else: phase = np.angle(overlap)
    return np.linalg.norm(d - u * np.exp(-1j * phase), ord=2)
    
ond_final = aligned_ond_full(U_target, Operator(qc_full).data)
t_final = qc_full.count_ops().get('t', 0) + qc_full.count_ops().get('tdg', 0)

print(f"FINAL: T={t_final}, OND={ond_final}")

if ond_final < 0.1 and t_final < 630:
    print("SUCCESS: Valid and Efficient!")
    qiskit.qasm2.dump(qc_full, "./solution_challenge_10.qasm")