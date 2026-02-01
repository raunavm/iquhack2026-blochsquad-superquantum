import numpy as np
from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit
import heapq
import time

a = (1+1j)/2
M_target = np.array([
    [1, 0, 0, 0],
    [0, 0, -a, a],
    [0, 1j, 0, 0],
    [0, 0, -a, -a]
], dtype=complex)

S = np.array([[1, 0], [0, 1j]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

single_gates = [
    ('h', H, 0.1), ('s', S, 0.1), ('sdg', S.conj().T, 0.1),
    ('t', T, 1), ('tdg', T.conj().T, 1)
]

ops = []
for name, mat, cost in single_gates:
    op0 = np.kron(np.eye(2), mat) 
    ops.append((f"{name}(0)", op0, cost))
    
    op1 = np.kron(mat, np.eye(2))
    ops.append((f"{name}(1)", op1, cost))

CX01 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]) # C=0, T=1?
ops.append(("cx(0,1)", CX01, 0.1))

CX10 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
ops.append(("cx(1,0)", CX10, 0.1))

def aligned_dist(U, V):
    tr = np.trace(U.conj().T @ V)
    return 1 - np.abs(tr) / 4.0 # Normalized 1 - F


def exact_search(target, max_t=4):
    counter = 0
    pq = []
    heapq.heappush(pq, (0, 0, 0, np.eye(4, dtype=complex), []))
    
    iters = 0
    start = time.time()
    
    while pq:
        _, t_cnt, _, curr_u, circ = heapq.heappop(pq) # Unpack counter
        iters += 1
        
        d = aligned_dist(target, curr_u)
        if d < 1e-4:
            print(f"Found! T={t_cnt}, Circ={circ}")
            return circ
        
        if t_cnt >= max_t:
            continue
            
        last_op = circ[-1] if circ else ""
        
        for name, mat, cost in ops:
            if "t" in name and "tdg" in last_op and name[:3]==last_op[:3]: continue
            if "h" in name and "h" in last_op and name==last_op: continue 
            
            new_u = mat @ curr_u 
            new_t = t_cnt + cost
            
            if new_t > max_t: continue

            new_circ = circ + [name]
            
            h = aligned_dist(target, new_u)
            priority = new_t + h * 1.0 
            
            counter += 1
            heapq.heappush(pq, (priority, new_t, counter, new_u, new_circ))
            
        if iters % 10000 == 0:
            print(f"Iter {iters}, PQ {len(pq)}, Best Dist {d}")
            if time.time() - start > 60: 
                break
    return None

print("Starting Search T<=4...")
exact_search(M_target, max_t=4)