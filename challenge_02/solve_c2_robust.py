import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2
from qiskit.quantum_info import Operator
import heapq

# Target: Ry(pi/14)
theta = np.pi / 14
target_u = np.array([
    [np.cos(theta/2), -np.sin(theta/2)],
    [np.sin(theta/2), np.cos(theta/2)]
], dtype=complex)

gate_names = ['h', 's', 'sdg', 't', 'tdg', 'x', 'z', 'y']
gate_costs = {'h':0, 's':0, 'sdg':0, 't':1, 'tdg':1, 'x':0, 'z':0, 'y':0}

S = np.array([[1, 0], [0, 1j]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

mats_dict = {
    'h': H, 's': S, 'sdg': S.conj().T, 
    't': T, 'tdg': T.conj().T,
    'x': X, 'z': Z, 'y': Y
}

def aligned_dist(U, V):
    tr = np.trace(U.conj().T @ V)
    return np.sqrt(np.abs(2 - 2 * np.abs(tr) / 2))

def beam_search(max_t=9, width=5000):
    # (dist, t, seq, mat)
    beam = [(aligned_dist(target_u, I), 0, [], I)]
    
    best_sol = (100.0, None)
    
    for depth in range(30):
        next_beam = []
        print(f"Depth {depth}, Beam size {len(beam)}, Best D={best_sol[0]}")
        
        for d, t, seq, mat in beam:
            # Expand
            for g_name in gate_names:
                # Pruning
                if seq:
                    pass # TODO: Add more pruning
                
                n_t = t + gate_costs[g_name]
                if n_t > max_t: continue
                
                n_mat = mats_dict[g_name] @ mat
                n_d = aligned_dist(target_u, n_mat)
                n_seq = seq + [g_name]
                
                next_beam.append((n_d, n_t, n_seq, n_mat))
                
                if n_d < best_sol[0]:
                    best_sol = (n_d, n_seq)
                    print(f"  New Best: T={n_t}, D={n_d}")
        
        # Select best
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:width]
        
        if not beam: break
        
    return best_sol

print("Starting Robust Beam Search...")
d, seq = beam_search(max_t=9, width=10000)

print(f"Best: T={len([g for g in seq if 't' in g])}, D={d}")

if d < 0.013:
    print("SUCCESS: Target Met (Single Err < 0.013)")
    # Generate qasm ...
else:
    print("Failed to meet target.")
