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

def aligned_dist(U, V):
    tr = np.trace(U.conj().T @ V)
    return np.sqrt(np.abs(2 - 2 * np.abs(tr) / 2)) 

def stochastic_hill_climb(target_u, max_t=80):
    best_circ = []
    best_u = I
    best_dist = aligned_dist(target_u, I)
    
    gs = gate_set
    n_gates = len(gs)
    
    print("  Initializing with Random Search (Depth 20)...")
    for _ in range(50000):
        curr_u = I
        curr_circ = []
        depth = np.random.randint(5, 20)
        for _ in range(depth):
            idx = np.random.randint(n_gates)
            name, mat, cost = gs[idx]
            curr_u = mat @ curr_u
            curr_circ.append(name)
        
        d = aligned_dist(target_u, curr_u)
        if d < best_dist:
            best_dist = d
            best_circ = curr_circ
            best_u = curr_u
            if d < 0.15: break # Good start
            
    print(f"  Start Hill Climb: T={len([g for g in best_circ if 't' in g])}, D={best_dist}")
    
    max_steps = 10000
    for step in range(max_steps):
        if best_dist < 0.07:
            print("  Converged!")
            return best_circ, best_dist
            
        current_t = len([g for g in best_circ if 't' in g])
        if current_t >= max_t:
            print("  Max T reached in HC.")
            return best_circ, best_dist
            
        ext_len = np.random.randint(1, 6)
        cand_u = best_u
        cand_circ = list(best_circ)
        
        
        for _ in range(ext_len):
            idx = np.random.randint(n_gates)
            name, mat, cost = gs[idx]
            cand_u = mat @ cand_u
            cand_circ.append(name)
            
        d = aligned_dist(target_u, cand_u)
        
        if d < best_dist - 0.0001: # Epsilon improvement
            best_dist = d
            best_circ = cand_circ
            best_u = cand_u
            
    print("  Hill Climb finished (steps limit).")
    return best_circ, best_dist

qc_approx_full = QuantumCircuit(2)
print("Synthesizing Single Qubit Gates (Hill Climb)...")

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
        
        circ, dist = stochastic_hill_climb(u_target, max_t=80) 
        
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
qiskit.qasm2.dump(qc_approx_full, "/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_10.qasm")