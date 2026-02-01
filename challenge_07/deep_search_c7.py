import numpy as np
import random
import heapq
from qiskit.circuit.library import UGate

targets = [
    UGate(np.pi/2, 0.27221, 4.3799).to_matrix(),
    UGate(2.2215, 4.0137, 1.5114).to_matrix(),
    UGate(0.90157, 1.7606, 0).to_matrix()
]

ops = {
    'h': 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex),
    's': np.array([[1, 0], [0, 1j]], dtype=complex),
    't': np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex),
    'sdg': np.conj(np.array([[1, 0], [0, 1j]], dtype=complex)).T,
    'tdg': np.conj(np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)).T
}
op_keys = list(ops.keys())

def generate_database(depth, size_limit=200000):
    db = {}
    pq = []
    q = [ ([], np.eye(2, dtype=complex), 0) ] 
    
    count = 0
    while q and count < size_limit:
        seq, mat, t = q.pop(0)
        
        pass 
    return db

def random_search(target, iterations=200000, max_len=40):
    best_ond = 1.0
    best_seq = []
    
    for _ in range(iterations):
        u = np.eye(2, dtype=complex)
        seq = []
        l = random.randint(15, 60)
        for _ in range(l):
             k = random.choice(op_keys)
             u = ops[k] @ u
             seq.append(k)
             
        d = target
        m = u
        overlap = np.trace(d.conj().T @ m)
        phase = np.angle(overlap) if np.abs(overlap) > 1e-9 else 0
        ond = np.linalg.norm(d - m * np.exp(-1j * phase), ord=2)
        
        if ond < best_ond:
            best_ond = ond
            best_seq = seq
            
    return best_seq, best_ond

print("Starting Deep Search...")
final_seqs = []
total_t = 0
for i, target in enumerate(targets):
    print(f"Target {i}...")
    seq, ond = random_search(target, iterations=400000, max_len=60)
    t = seq.count('t') + seq.count('tdg')
    print(f"  Best: T={t}, OND={ond}")
    total_t += t
    final_seqs.append(seq)

print(f"Total T: {total_t}")

if total_t < 300:
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    
    
    
    for g in final_seqs[0]: getattr(qc, g)(0)
    
    for g in final_seqs[2]: getattr(qc, g)(1)
    
    qc.cx(1, 0)
    
    for g in final_seqs[1]: getattr(qc, g)(0)
    
    import qiskit.qasm2
    qiskit.qasm2.dump(qc, "/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_7_efficient.qasm")
    print("Saved efficient solution.")