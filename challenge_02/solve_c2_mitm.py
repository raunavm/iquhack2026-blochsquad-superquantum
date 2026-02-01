import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2
from qiskit.quantum_info import Operator
from scipy.spatial import KDTree

# Target: Ry(pi/14)
theta = np.pi / 14
target_u = np.array([
    [np.cos(theta/2), -np.sin(theta/2)],
    [np.sin(theta/2), np.cos(theta/2)]
], dtype=complex)

gate_names = ['h', 's', 'sdg', 't', 'tdg', 'x', 'z', 'y']
gate_costs = {'h':0, 's':0, 'sdg':0, 't':1, 'tdg':1, 'x':0, 'z':0, 'y':0}
# Reduced set for efficiency?
# Ry is real. Maybe stick to real gates? H, Z, X, T? No T is complex phases.
# But Ry result is real.
# H, T, Tdg, S, Sdg generate unitary group.

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

def generate_states(max_t=5, max_depth=12):
    # BFS
    # state: (mat, t_count, seq)
    states = []
    queue = [(I, 0, [])]
    
    seen = set() # Hash of matrix?
    
    # Simple processing
    import collections
    q = collections.deque([(I, 0, [])])
    
    while q:
        mat, t, seq = q.popleft()
        
        # Store
        states.append((mat, t, seq))
        
        # Expand
        if t >= max_t: continue # Can add Clifford but limits depth
        
        if len(seq) >= max_depth: continue
        
        for g in gate_names:
            # Pruning
            if seq and seq[-1] == g: continue # heuristic
            
            nt = t + gate_costs[g]
            if nt > max_t: continue
            
            nmat = mats_dict[g] @ mat
            nseq = seq + [g]
            
            q.append((nmat, nt, nseq))
            
    return states

print("Generating Forward States (T<=5)...")
fwd_states = generate_states(max_t=5, max_depth=10)
print(f"Forward: {len(fwd_states)}")

print("Generating Backward States (T<=4)...")
# Backward: Target @ U_inv ? 
# We want Fwd = Bwd.
# Fwd = Gz...G1.
# Bwd = Gk...Gm U_target_inv? NO.
# U_total = Gk... G1.
# U_total = U_target.
# Gk...G1 = U_target.
# Split: G_left ... = U_target ... G_right_inv.
# Let's say Left is T=5, Right is T=4.
# Left = U_target @ Right_inv.
# So we generate Right states, invert them, then multiply U_target by Right_inv.
bwd_raw = generate_states(max_t=4, max_depth=8)
bwd_states = []
for mat, t, seq in bwd_raw:
    # Right_inv = mat_dag
    # Target @ Right_inv
    val = target_u @ mat.conj().T
    bwd_states.append((val, t, seq))
print(f"Backward: {len(bwd_states)}")

# Convert to vectors for KDTree
# Unitary 2x2 -> 4 complex -> 8 real.
def to_vec(U):
    return np.concatenate([U.flatten().real, U.flatten().imag])

print("Building KDTree...")
fwd_vecs = np.array([to_vec(x[0]) for x in fwd_states])
bwd_vecs = np.array([to_vec(x[0]) for x in bwd_states])

tree = KDTree(fwd_vecs)

print("Querying...")
# Query bwd against fwd
# Need dist < epsilon.
# Start with nearest neighbor.
dists, indices = tree.query(bwd_vecs, k=1)

min_idx = np.argmin(dists)
min_dist = dists[min_idx]
best_fwd_idx = indices[min_idx]

fwd_best = fwd_states[best_fwd_idx]
bwd_best = bwd_states[min_idx]

# Total T
total_t = fwd_best[1] + bwd_best[1]
print(f"Best Match: T={total_t}, EuclDist={min_dist}")

# Reconstruct
# U ~ Fwd @ Bwd_orig.
# Fwd is fwd_best[2].
# Bwd_orig. we stored (Target @ Bwd_inv).
# Wait.
# If Fwd ~ Target @ Bwd_inv.
# Then Fwd @ Bwd ~ Target.
# Bwd_orig is just bwd_raw[min_idx].
bwd_orig = bwd_raw[min_idx]
# Sequence: Fwd_seq + inverse(Bwd_seq).
# Bwd_seq built G1..Gk.
# Matrix was Gk..G1.
# Total = Fwd @ Bwd_inv? No.
# left = target @ right_inv.
# left @ right = target.
# Left matrix is G_L ... G_1.
# Right matrix is G_R ... G_1.
# Total = Left @ Right_inv? No.
# If Left = G_L...
# Right_inv = (G_R...)^dag = ... G_R^dag.
# So Left @ Right_inv = (G_L...) (G_1^dag ... G_R^dag).
# Yes.
inverse_bwd_seq = []
# Inverse of [g1, g2] is [g2_dag, g1_dag].
inv_map = {'h':'h', 's':'sdg', 'sdg':'s', 't':'tdg', 'tdg':'t', 'x':'x', 'y':'y', 'z':'z'}
for g in reversed(bwd_orig[2]):
    inverse_bwd_seq.append(inv_map[g])
    
full_seq = fwd_best[2] + inverse_bwd_seq

print(f"Full Seq: {full_seq}")

# Verify
qc = QuantumCircuit(2)
# Ry(t/2)
for g in full_seq:
    if g=='h': qc.h(1)
    elif g=='s': qc.s(1)
    elif g=='sdg': qc.sdg(1)
    elif g=='t': qc.t(1)
    elif g=='tdg': qc.tdg(1)
    elif g=='x': qc.x(1)
    elif g=='z': qc.z(1)
    elif g=='y': qc.y(1)

qc.cx(0, 1)

# Ry(-t/2)
for g in reversed(full_seq):
    if g=='h': qc.h(1)
    elif g=='s': qc.sdg(1)
    elif g=='sdg': qc.s(1)
    elif g=='t': qc.tdg(1)
    elif g=='tdg': qc.t(1)
    elif g=='x': qc.x(1)
    elif g=='z': qc.z(1)
    elif g=='y': qc.y(1)
qc.cx(0, 1)

# Calc OND
target_cry = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, np.cos(np.pi/14), -np.sin(np.pi/14)],
    [0, 0, np.sin(np.pi/14), np.cos(np.pi/14)]
], dtype=complex)

final_op = Operator(qc).data
def aligned_ond_full(U, V):
    overlap = np.trace(U.conj().T @ V)
    phase = np.angle(overlap)
    return np.linalg.norm(U - V * np.exp(-1j * phase), ord=2)

final_dist = aligned_ond_full(target_cry, final_op)
final_t = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)

print(f"FINAL C2: T={final_t}, Dist={final_dist}")

if final_dist < 0.027:
    print("SUCCESS")
    qiskit.qasm2.dump(qc, "challenge_02/solution_challenge_2_opt_mitm.qasm")
else:
    print("Optimization partial.")
