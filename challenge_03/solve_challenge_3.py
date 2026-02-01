import numpy as np
from qiskit.quantum_info import Operator
import itertools
from qiskit import QuantumCircuit
import qiskit.qasm2
import heapq

def aligned_ond(U, V):
    d = U
    u = V
    overlap = np.trace(d.conj().T @ u)
    if np.abs(overlap) < 1e-9:
        phase = 0
    else:
        phase = np.angle(overlap)
    u_aligned = u * np.exp(-1j * phase)
    return np.linalg.norm(d - u_aligned, ord=2)

target_phi = np.pi / 7
U_target = np.diag([np.exp(1j * target_phi), np.exp(-1j * target_phi), np.exp(-1j * target_phi), np.exp(1j * target_phi)])

print(f"Target: exp(i pi/7 ZZ)")

bench_phi = np.pi / 8
U_bench = np.diag([np.exp(1j * bench_phi), np.exp(-1j * bench_phi), np.exp(-1j * bench_phi), np.exp(1j * bench_phi)])
ond_bench = aligned_ond(U_target, U_bench)
print(f"Benchmark T=1 (pi/8): OND={ond_bench}")


print("Searching for Rz(-2 pi/7) approximations...")
target_angle = - 2 * np.pi / 7

target_rz = np.diag([np.exp(-1j * target_angle/2), np.exp(1j * target_angle/2)])

ops = {
    'H': 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex),
    'S': np.array([[1, 0], [0, 1j]], dtype=complex),
    'T': np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex),
}
ops['Sdg'] = np.conj(ops['S']).T
ops['Tdg'] = np.conj(ops['T']).T

pq = []
heapq.heappush(pq, (1.0, 0, 0, 0, [], np.eye(2, dtype=complex).tolist()))

best_ond = 0.0561
found_better = False

max_t = 10
nodes = 0
while pq and nodes < 200000:
    nodes += 1
    ond, length, t, _, seq, mat_list = heapq.heappop(pq)
    
    if ond < best_ond - 1e-5 and t <= max_t:
        print(f"FOUND BETTER: T={t}, OND={ond:.6f}, Seq={seq}")
        best_ond = ond
        found_better = True
        
    if t >= max_t: continue
    if length > 12: continue # Depth limit

    mat = np.array(mat_list, dtype=complex)
    last_op = seq[-1] if seq else ""

    for name, op in ops.items():
        if name == 'H' and last_op == 'H': continue
        
        new_mat = op @ mat
        new_seq = seq + [name]
        new_t = t + (1 if 'T' in name else 0)
        
        new_ond = aligned_ond(target_rz, new_mat)
        
        heapq.heappush(pq, (new_ond, length+1, new_t, nodes, new_seq, new_mat.tolist()))

if found_better:
    print("Found solution!")
else:
    print("No better solution (T<=10) found.")