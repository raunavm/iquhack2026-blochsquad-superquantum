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

gate_mats = np.array([H, S, S.conj().T, T, T.conj().T]) # Shape (5, 2, 2)
gate_names = ['h', 's', 'sdg', 't', 'tdg']
gate_costs = np.array([0, 0, 0, 1, 1])

def aligned_dist_vec(Target, Batch):
    
    T_H = Target.conj().T
    
    Product = np.matmul(T_H, Batch)
    
    Tr = Product[:, 0, 0] + Product[:, 1, 1]
    
    Dist = np.sqrt(np.abs(2 - 2 * np.abs(Tr) / 2))
    return Dist

def vectorized_search(target_u, batch_size=20000, max_depth=30):
    states = np.tile(np.eye(2, dtype=complex), (batch_size, 1, 1))
    
    
    
    paths = np.full((batch_size, max_depth), -1, dtype=int)
    
    current_best_dist = 10.0
    current_best_seq = []
    
    for d in range(max_depth):
        choices = np.random.randint(5, size=batch_size)
        paths[:, d] = choices
        
        chosen_mats = gate_mats[choices]
        
        states = np.matmul(chosen_mats, states)
        
        dists = aligned_dist_vec(target_u, states)
        
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        if min_dist < current_best_dist:
            current_best_dist = min_dist
            raw_seq_idx = paths[min_idx, :d+1]
            current_best_seq = [gate_names[i] for i in raw_seq_idx]
            
            if min_dist < 0.07: # Threshold for residual? 
                pass
                
    return current_best_seq, current_best_dist, current_best_dist < 0.2

def iterative_vector_search(target_u, max_total_t=600):
    full_circ = []
    current_u = I
    
    for stage in range(10): # Stages
        residual = current_u.conj().T @ target_u
        
        dist_now = aligned_dist_vec(target_u, current_u.reshape(1,2,2))[0]
        t_now = len([g for g in full_circ if 't' in g])
        print(f"  Stage {stage}: D={dist_now}, T={t_now}")
        
        if dist_now < 0.07:
            print("  Converged!")
            return full_circ, dist_now
            
        if t_now >= max_total_t:
            print("  Max T.")
            return full_circ, dist_now

        depth = 15 if stage == 0 else 30
        batch = 50000
        
        seq, d_res, improved = vectorized_search(residual, batch_size=batch, max_depth=depth)
        
        full_circ.extend(seq)
        
        for g in seq:
            idx = gate_names.index(g)
            mat = gate_mats[idx]
            current_u = mat @ current_u
            
    return full_circ, dist_now

qc_approx_full = QuantumCircuit(2)
print("Synthesizing Single Qubit Gates (Vectorized)...")

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
        
        circ, dist = iterative_vector_search(u_target)
        
        t_cnt = len([g for g in circ if 't' in g])
        print(f"  Result: T={t_cnt}, D={dist}")

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
qiskit.qasm2.dump(qc_approx_full, "./solution_challenge_10.qasm")