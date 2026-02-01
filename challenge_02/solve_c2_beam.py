import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2
from qiskit.quantum_info import Operator
import heapq

# Target: Ry(pi/14)
# theta = pi/14
# Ry(theta) = exp(-i theta/2 Y) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
theta = np.pi / 14
target_u = np.array([
    [np.cos(theta/2), -np.sin(theta/2)],
    [np.sin(theta/2), np.cos(theta/2)]
], dtype=complex)

# Gate Set
S = np.array([[1, 0], [0, 1j]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

gate_mats = [H, S, S.conj().T, T, T.conj().T, X, Z, Y]
gate_names = ['h', 's', 'sdg', 't', 'tdg', 'x', 'z', 'y']
gate_costs = [0, 0, 0, 1, 1, 0, 0, 0]

def aligned_dist(U, V):
    tr = np.trace(U.conj().T @ V)
    return np.sqrt(np.abs(2 - 2 * np.abs(tr) / 2))

def beam_search(target, max_t=9, beam_width=5000):
    # (dist, t, seq, mat)
    # Priority Queue ordered by distance
    pq = []
    heapq.heappush(pq, (aligned_dist(target, I), 0, [], I))
    
    # Store visited to pruning?
    # Simple beam: keep top K at each depth?
    # Or just generic priority queue search (Best First).
    # With T cost limit.
    
    best_sol = (100.0, None)
    
    visited_hashes = set()
    
    # Let's do strict Beam Search by T-count layers?
    # Or Best First.
    # Best First is good for finding optimal T.
    
    # Actually, we have a fixed budget. We want minimal Distance.
    # So Layered Beam Search by Gate Depth is better.
    
    current_beam = [(aligned_dist(target, I), 0, [], I)]
    
    # We loop deeper and deeper
    for depth in range(30): # gate depth
        next_beam = []
        print(f"Depth {depth}, Beam size {len(current_beam)}, Best Dist {best_sol[0]}")
        
        for d, t, seq, mat in current_beam:
            if t > max_t: continue
            
            # Prune if bad?
            # Expand
            for i, g_mat in enumerate(gate_mats):
                # Optimize: Don't do inverse of last gate.
                if len(seq) > 0:
                    last = seq[-1]
                    curr = gate_names[i]
                    if last == 'h' and curr == 'h': continue
                    if last == 'x' and curr == 'x': continue
                    # ... more pruning rules possible
                
                n_t = t + gate_costs[i]
                if n_t > max_t: continue
                    
                n_mat = g_mat @ mat
                
                # Check dist
                n_d = aligned_dist(target, n_mat)
                n_seq = seq + [gate_names[i]]
                
                if n_d < best_sol[0]:
                    best_sol = (n_d, n_seq)
                    print(f"  New Best: T={n_t}, D={n_d}")
                    if n_d < 0.01: # Early exit target
                        pass
                
                next_beam.append((n_d, n_t, n_seq, n_mat))
                
        # Prune Beam
        # Sort by distance
        next_beam.sort(key=lambda x: x[0])
        
        # Deduplicate roughly? O(N^2) is slow.
        # Just take top K
        current_beam = next_beam[:beam_width]
        
        if not current_beam: break
        
    return best_sol

print("Searching for Ry(pi/14)...")
best_dist, best_seq = beam_search(target_u, max_t=9, beam_width=20000) # Increased width

print(f"Best Sequence: {best_seq}")
print(f"T-count: {len([g for g in best_seq if 't' in g])}")
print(f"Dist: {best_dist}")

# Reconstruct Full Circuit
# CRy(pi/7) ~ Ry(pi/14) on q1 -> CX 0,1 -> Ry(-pi/14) on q1 -> CX 0,1
# Need Ry(-pi/14). = Ry(pi/14) dagger.
# Just reverse sequence and conjugate gates.
# T -> Tdg, Tdg -> T, S -> Sdg, H->H...

qc = QuantumCircuit(2)
# Ry(theta/2) on q[1]
for g in best_seq:
    if g=='h': qc.h(1)
    elif g=='s': qc.s(1)
    elif g=='sdg': qc.sdg(1)
    elif g=='t': qc.t(1)
    elif g=='tdg': qc.tdg(1)
    elif g=='x': qc.x(1)
    elif g=='y': qc.y(1)
    elif g=='z': qc.z(1)

qc.cx(0, 1)

# Ry(-theta/2) on q[1] = Inverse of Ry(theta/2)
# Since unitary, inverse is reverse order, dagger.
for g in reversed(best_seq):
    if g=='h': qc.h(1)
    elif g=='s': qc.sdg(1)
    elif g=='sdg': qc.s(1)
    elif g=='t': qc.tdg(1)
    elif g=='tdg': qc.t(1)
    elif g=='x': qc.x(1)
    elif g=='y': qc.y(1) # Y dag = Y
    elif g=='z': qc.z(1)

qc.cx(0, 1)

# Check total OND
target_cry = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, np.cos(np.pi/14), -np.sin(np.pi/14)],
    [0, 0, np.sin(np.pi/14), np.cos(np.pi/14)]
], dtype=complex) # approx Ry(pi/7) on 2nd block

# My construction: 
# (I x Ry(t/2)) CX (I x Ry(-t/2)) CX
# = (I x A) . |0><0|xI + |1><1|xX . (I x Adg) . |0><0|xI + |1><1|xX
# Control 0: I x A . I . I x Adg . I = I x (A Adg) = I. Correct.
# Control 1: I x A . X . I x Adg . X = I x (A X Adg X)
# We want A X Adg X = Ry(theta).
# A = Ry(theta/2).
# Ry(theta/2) X Ry(-theta/2) X = Ry(theta/2) Ry(theta/2) = Ry(theta). Correct.

# Calculate dist
final_op = Operator(qc).data
def aligned_ond_full(U, V):
    overlap = np.trace(U.conj().T @ V)
    phase = np.angle(overlap)
    return np.linalg.norm(U - V * np.exp(-1j * phase), ord=2)

full_dist = aligned_ond_full(target_cry, final_op)
t_count = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)

print(f"FINAL C2: T={t_count}, Dist={full_dist}")

qiskit.qasm2.dump(qc, "challenge_02/solution_challenge_2_opt.qasm")
