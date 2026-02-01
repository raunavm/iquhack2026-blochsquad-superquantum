import numpy as np
import random
from qiskit import QuantumCircuit
import qiskit.qasm2

def aligned_ond(U, V):
    d = U
    u = V
    overlap = np.trace(d.conj().T @ u)
    if np.abs(overlap) < 1e-9:
        phase = np.angle(overlap) # 0?
        phase = 0
    else:
        phase = np.angle(overlap)
    u_aligned = u * np.exp(-1j * phase)
    return np.linalg.norm(d - u_aligned, ord=2)

target_angle = - 2 * np.pi / 7
target_rz = np.diag([np.exp(-1j * target_angle/2), np.exp(1j * target_angle/2)])

best_ond = 0.05609
print(f"Goal: Beat OND {best_ond}")

ops = {
    'h': 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex),
    's': np.array([[1, 0], [0, 1j]], dtype=complex),
    't': np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex),
}
ops['sdg'] = np.conj(ops['s']).T
ops['tdg'] = np.conj(ops['t']).T
op_keys = list(ops.keys())

for i in range(200000):
    u = np.eye(2, dtype=complex)
    seq = []
    
    length = random.randint(5, 25)
    for _ in range(length):
        k = random.choice(op_keys)
        u = ops[k] @ u
        seq.append(k)
        
    ond = aligned_ond(target_rz, u)
    t_count = seq.count('t') + seq.count('tdg')
    
    if ond < best_ond - 1e-4:
        print(f"FOUND BETTER: OND={ond:.6f}, T={t_count}, Seq={seq}")
        qc = QuantumCircuit(2)
        
        name_map = {'h': 'h', 's': 's', 'sdg': 'sdg', 't': 't', 'tdg': 'tdg'}
        inv_map = {'h': 'h', 's': 'sdg', 'sdg': 's', 't': 'tdg', 'tdg': 't'}
        
        qc.cx(0, 1)
        
        for g in seq: getattr(qc, name_map[g])(1)
        
        qc.cx(0, 1)
        
        
        filename = f"./solution_challenge_3_better.qasm"
        qiskit.qasm2.dump(qc, filename)
        
        best_ond = ond