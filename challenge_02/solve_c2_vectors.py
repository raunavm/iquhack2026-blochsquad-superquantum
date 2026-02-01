import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2
from qiskit.quantum_info import Operator
import time

theta = np.pi / 14
target_u = np.array([
    [np.cos(theta/2), -np.sin(theta/2)],
    [np.sin(theta/2), np.cos(theta/2)]
], dtype=complex)

gate_names = ['h', 's', 'sdg', 't', 'tdg', 'x', 'z']
gate_costs = np.array([0, 0, 0, 1, 1, 0, 0])

S = np.array([[1, 0], [0, 1j]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Stack for indexing
gate_mats_stack = np.stack([H, S, S.conj().T, T, T.conj().T, X, Z])

def vector_search(total_trials=2000000, chunk_size=50000, depth=50):
    best_sol = (10.0, None)
    
    num_chunks = total_trials // chunk_size
    
    for c in range(num_chunks):
        if c % 5 == 0: print(f"Processing chunk {c}/{num_chunks}...")
        
        # Random paths
        paths = np.random.randint(len(gate_names), size=(chunk_size, depth))
        
        # Init state
        # (Chunk, 2, 2)
        states = np.tile(I, (chunk_size, 1, 1))
        t_counts = np.zeros(chunk_size, dtype=int)
        
        # Evolution
        for d in range(depth):
            indices = paths[:, d]
            
            # Select matrices
            # fancy indexing
            chosen = gate_mats_stack[indices] # (Chunk, 2, 2)
            
            # Update: New = G @ Old
            # Batch matmul: (Chunk, 2, 2) @ (Chunk, 2, 2)
            # Use einsum for explicit safety
            # states = np.einsum('nij,njk->nik', chosen, states)
            states = np.matmul(chosen, states)
            
            # Update T
            costs = gate_costs[indices]
            t_counts += costs
            
            # Check conditions
            # Only check if T <= 9?
            # Or check always and sort by T?
            # Let's check D for ALL, then filter by T.
            
            # Calculate dist
            # tr(Target_dag @ State)
            # (2, 2) @ (Chunk, 2, 2) -> (Chunk, 2, 2)
            # elementwise? No.
            # explicit:
            u_dag = target_u.conj().T
            prods = np.matmul(u_dag, states) # (Chunk, 2, 2)
            
            trs = prods[:, 0, 0] + prods[:, 1, 1]
            dists = np.sqrt(np.abs(2 - 2 * np.abs(trs) / 2))
            
            # Filter
            # Single T < 10 (Total < 20)
            # Single Dist < 0.0132 (Total < 0.027 approx 2.05*D)
            
            # Look for ANY valid solution better than competitor
            mask_t = t_counts <= 9
            mask_d = dists < 0.0133
            
            combined = mask_t & mask_d
            if np.any(combined):
                valid_indices = np.where(combined)[0]
                best_idx = valid_indices[np.argmin(dists[valid_indices])]
                
                final_d = dists[best_idx]
                final_t = t_counts[best_idx]
                
                print(f"FOUND! T={final_t}, D={final_d}")
                return final_d, paths[best_idx][:d+1]
                
            # Keep best overall for logging
            min_idx = np.argmin(dists)
            if dists[min_idx] < best_sol[0]:
                best_sol = (dists[min_idx], paths[min_idx][:d+1])
                # print(f"  Best seen: T={t_counts[min_idx]}, D={dists[min_idx]}")

    return best_sol

print("Starting Vector Search...")
start = time.time()
d, path_idx = vector_search()
print(f"Time: {time.time()-start:.2f}s")
print(f"Best: D={d}")

if path_idx is not None and d < 0.0133:
    seq = [gate_names[i] for i in path_idx]
    print(f"Seq: {seq}")
    
    qc = QuantumCircuit(2)
    # Ry(t/2)
    for g in seq:
        if g=='h': qc.h(1)
        elif g=='s': qc.s(1)
        elif g=='sdg': qc.sdg(1)
        elif g=='t': qc.t(1)
        elif g=='tdg': qc.tdg(1)
        elif g=='x': qc.x(1)
        elif g=='z': qc.z(1)
    
    qc.cx(0, 1)
    
    # Ry(-t/2)
    for g in reversed(seq):
        if g=='h': qc.h(1)
        elif g=='s': qc.sdg(1)
        elif g=='sdg': qc.s(1)
        elif g=='t': qc.tdg(1)
        elif g=='tdg': qc.t(1)
        elif g=='x': qc.x(1)
        elif g=='z': qc.z(1)
    
    qc.cx(0, 1)
    
    final_t = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)
    
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
    
    print(f"FINAL C2: T={final_t}, Dist={dist}")
    qiskit.qasm2.dump(qc, "challenge_02/solution_challenge_2_opt_vec.qasm")
else:
    print("No valid solution found.")
