import numpy as np
from qiskit.quantum_info import Operator
import itertools

b = (-1+1j)/2
c = (1+1j)/2
d = (-1-1j)/2

M = np.array([
    [1, 0, 0, 0],
    [0, 0, b, c],
    [0, 1j, 0, 0],
    [0, 0, b, d]
], dtype=complex)

print("Initial M:")
print(np.round(M, 3))

S = np.array([[1, 0], [0, 1j]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
CX01 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
CX10 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])

gate_pool = [
    ("h(0)", np.kron(np.eye(2), H)),
    ("h(1)", np.kron(H, np.eye(2))),
    ("s(0)", np.kron(np.eye(2), S)),
    ("s(1)", np.kron(S, np.eye(2))),
    ("cx01", CX01),
    ("cx10", CX10),
    ("x(0)", np.kron(np.eye(2), X)),
    ("x(1)", np.kron(X, np.eye(2))),
    ("swap", SWAP)
]

def score_matrix(mat):
    off_diag = np.sum(np.abs(mat - np.diag(np.diag(mat))))
    if off_diag < 1e-4:
        return 100 # Diagonal
    
    zeros = np.sum(np.abs(mat) < 1e-4)
    return zeros

curr_M = M.copy()
seq = []

for step in range(10): # Max 10 cliffords
    best_score = -1
    best_op = None
    best_name = ""
    
    if score_matrix(curr_M) == 100:
        print("Diagonalized!")
        break
        
    current_score = score_matrix(curr_M)
    
    for name, op in gate_pool:
        cand = op.conj().T @ curr_M # Inverse of Clifford peel?
        cand = op @ curr_M 
        
        s = score_matrix(cand)
        if s > current_score + 0.1: # Strict improvement
            if s > best_score:
                best_score = s
                best_op = op
                best_name = name
    
    if best_op is not None:
        curr_M = best_op @ curr_M
        seq.append(best_name)
        print(f"Step {step}: Applied {best_name}, Score {best_score}")
    else:
        print("Stuck.")
        break

print("Final Remainder (Diagonal?):")
print(np.round(curr_M, 3))
print("Sequence (Applied to Left):", seq)