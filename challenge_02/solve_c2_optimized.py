import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2
from qiskit.quantum_info import Operator
import time

# Target: Ry(pi/14)
theta = np.pi / 14
target_u = np.array([
    [np.cos(theta/2), -np.sin(theta/2)],
    [np.sin(theta/2), np.cos(theta/2)]
], dtype=complex)

# Metrics
def aligned_dist(U, V):
    tr = np.trace(U.conj().T @ V)
    return np.sqrt(np.abs(2 - 2 * np.abs(tr) / 2))

# Gate Set
S = np.array([[1, 0], [0, 1j]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

gate_mats = np.array([H, S, S.conj().T, T, T.conj().T, X, Z]) 
gate_names = ['h', 's', 'sdg', 't', 'tdg', 'x', 'z']
gate_costs = np.array([0, 0, 0, 1, 1, 0, 0])

def random_search(target, max_t=9, batch_size=500000, depth=40):
    best_sol = (10.0, None)
    
    # Vectorized
    # State: (Batch, 2, 2)
    # T_counts: (Batch,)
    
    # Init
    current = np.tile(I, (1, 1, 1)) # (1, 2, 2)
    current_t = np.array([0])
    current_seq_indices = np.zeros((1, 0), dtype=int)
    
    # Iterative? No, brute force paths.
    
    # Attempt: Generate random paths
    # (Batch, Depth)
    paths = np.random.randint(len(gate_names), size=(batch_size, depth))
    
    # Eval
    # Accumulate matrices
    mats = np.tile(I, (batch_size, 2, 2))
    t_counts = np.zeros(batch_size, dtype=int)
    
    # Process gate by gate to catch early wins and prune T
    for d in range(depth):
        gate_indices = paths[:, d]
        
        # Look up matrices
        # gate_mats (7, 2, 2)
        # selected (Batch, 2, 2)
        selected_mats = gate_mats[gate_indices]
        
        # Multiply: New = G @ Old
        mats = np.matmul(selected_mats, mats)
        
        # Update T
        selected_costs = gate_costs[gate_indices]
        t_counts += selected_costs
        
        # Check alignment
        # Only check if T <= max_t
        mask = t_counts <= max_t
        
        if np.any(mask):
            # Calculate dist for valid ones
            valid_mats = mats[mask]
            
            # aligned_dist vectorized
            # trace(U' @ V)
            u_dag = target.conj().T
            prods = np.matmul(u_dag, valid_mats)
            trs = prods[:, 0, 0] + prods[:, 1, 1]
            dists = np.sqrt(np.abs(2 - 2 * np.abs(trs) / 2))
            
            min_idx_local = np.argmin(dists)
            min_d = dists[min_idx_local]
            
            if min_d < best_sol[0]:
                best_sol = (min_d, paths[mask][min_idx_local][:d+1])
                print(f"  New Best (Depth {d+1}): T={t_counts[mask][min_idx_local]}, D={min_d}")
                if min_d < 0.013:
                    return best_sol

    return best_sol

print("Random Search Ry(pi/14) with T<=9...")
best_d, best_path_idx = random_search(target_u, max_t=9, batch_size=2000000, depth=50)

# Convert path
if best_path_idx is not None:
    best_seq = [gate_names[i] for i in best_path_idx]
    print(f"Best Seq: {best_seq}")
    t_c = len([g for g in best_seq if 't' in g])
    print(f"Single T={t_c} Dist={best_d}")
    
    # Construct Full
    qc = QuantumCircuit(2)
    # Ry(t/2)
    for g in best_seq:
        if g=='h': qc.h(1)
        elif g=='s': qc.s(1)
        elif g=='sdg': qc.sdg(1)
        elif g=='t': qc.t(1)
        elif g=='tdg': qc.tdg(1)
        elif g=='x': qc.x(1)
        elif g=='z': qc.z(1)

    qc.cx(0, 1)

    # Ry(-t/2)
    for g in reversed(best_seq):
        if g=='h': qc.h(1)
        elif g=='s': qc.sdg(1)
        elif g=='sdg': qc.s(1)
        elif g=='t': qc.tdg(1)
        elif g=='tdg': qc.t(1)
        elif g=='x': qc.x(1)
        elif g=='z': qc.z(1)
    
    qc.cx(0, 1)

    t_final = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)
    
    # Calc dist
    target_cry = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.cos(np.pi/14), -np.sin(np.pi/14)],
        [0, 0, np.sin(np.pi/14), np.cos(np.pi/14)]
    ], dtype=complex)
    
    op_final = Operator(qc).data
    
    overlap = np.trace(target_cry.conj().T @ op_final)
    phase = np.angle(overlap)
    dist = np.linalg.norm(target_cry - op_final * np.exp(-1j * phase), ord=2)
    
    print(f"FINAL C2: T={t_final}, Dist={dist}")
    
    if dist < 0.027 and t_final < 20:
        print("SUCCESS! Beaten ElixirBros.")
        qiskit.qasm2.dump(qc, "challenge_02/solution_challenge_2_opt.qasm")
else:
    print("No solution found.")
