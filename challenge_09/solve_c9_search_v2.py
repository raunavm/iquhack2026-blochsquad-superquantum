import numpy as np
from qiskit.quantum_info import Operator
import heapq
import time

b = (-1+1j)/2
c = (1+1j)/2
d = (-1-1j)/2

M_target = np.array([
    [1, 0, 0, 0],
    [0, 0, b, c],
    [0, 1j, 0, 0],
    [0, 0, b, d]
], dtype=complex)

print("Target Matrix:")
print(M_target)

S = np.array([[1, 0], [0, 1j]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
CX01 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]) # Control 0 Target 1
CX10 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]) # Control 1 Target 0

single_gates = [
    ('h', H, 0.1), ('s', S, 0.1), ('sdg', S.conj().T, 0.1),
    ('t', T, 1), ('tdg', T.conj().T, 1)
]

ops = []
for name, mat, cost in single_gates:
    ops.append((f"{name}(0)", np.kron(np.eye(2), mat), cost))
    ops.append((f"{name}(1)", np.kron(mat, np.eye(2)), cost))

ops.append(("cx(0,1)", CX01, 0.1))
ops.append(("cx(1,0)", CX10, 0.1))

def aligned_dist(U, V):
    tr = np.trace(U.conj().T @ V)
    return 1 - np.abs(tr) / 4.0

def exact_search(target, max_t=4):
    counter = 0
    pq = []
    heapq.heappush(pq, (0, 0, 0, np.eye(4, dtype=complex), []))
    
    iters = 0
    start = time.time()
    
    while pq:
        _, t_cnt, _, curr_u, circ = heapq.heappop(pq)
        iters += 1
        
        d = aligned_dist(target, curr_u)
        if d < 1e-6:
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

            if len(circ) > 12: continue 

            new_circ = circ + [name]
            
            h = aligned_dist(target, new_u)
            priority = new_t + h # A*
            
            counter += 1
            heapq.heappush(pq, (priority, new_t, counter, new_u, new_circ))
            
        if iters % 10000 == 0:
            print(f"Iter {iters}, PQ {len(pq)}, Best Dist {d}")
            if time.time() - start > 120: 
                break
    return None

exact_search(M_target, max_t=4.5)