import numpy as np
from qiskit.quantum_info import Operator, random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit.circuit.library import CXGate
from qiskit import QuantumCircuit, transpile
import qiskit.qasm2
import heapq

seed = 42
U_target = random_unitary(4, seed=seed).data
kak_decomp = TwoQubitBasisDecomposer(gate=CXGate(), euler_basis='U3')
qc_kak = kak_decomp(U_target)

S = np.array([[1, 0], [0, 1j]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
I = np.eye(2, dtype=complex)
gate_mats = np.array([H, S, S.conj().T, T, T.conj().T]) 
gate_names = ['h', 's', 'sdg', 't', 'tdg']
gate_costs = np.array([0, 0, 0, 1, 1])

def aligned_dist(U, V):
    tr = np.trace(U.conj().T @ V)
    return np.sqrt(np.abs(2 - 2 * np.abs(tr) / 2)) 

def beam_search_base(target_u, max_t=12, width=1000):
    beam = [(aligned_dist(target_u, I), 0, [], I)]
    
    best_base = (10.0, [], I)
    
    for step in range(15): # Depth steps
        next_beam = []
        for d, t, seq, mat in beam:
            for i in range(5):
                n_mat = gate_mats[i] @ mat
                n_t = t + gate_costs[i]
                if n_t > max_t: continue
                
                n_seq = seq + [gate_names[i]]
                n_d = aligned_dist(target_u, n_mat)
                
                next_beam.append((n_d, n_t, n_seq, n_mat))
                
                if n_d < best_base[0]:
                    best_base = (n_d, n_seq, n_mat)
        
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:width]
        
        if not beam: break
        
    return best_base

try:
    skd = SolovayKitaev(recursion_degree=1)
    from qiskit.transpiler import PassManager
    pm = PassManager(skd)
except:
    pm = None

def solve_gate(u_gate):
    dist, base_seq, base_mat = beam_search_base(u_gate, max_t=12, width=2000)
    print(f"    Base T={len([g for g in base_seq if 't' in g])}, D={dist}")
    
    residual = base_mat.conj().T @ u_gate
    
    qc_res = QuantumCircuit(1)
    qc_res.unitary(residual, 0)
    
    qc_corr = pm.run(qc_res)
    
    full_circ = QuantumCircuit(1)
    for g in base_seq:
        if g=='h': full_circ.h(0)
        elif g=='s': full_circ.s(0)
        elif g=='sdg': full_circ.sdg(0)
        elif g=='t': full_circ.t(0)
        elif g=='tdg': full_circ.tdg(0)
        
    full_circ.append(qc_corr.to_instruction(), [0])
    
    full_circ = transpile(full_circ, basis_gates=['h','s','sdg','t','tdg'])
    
    t_cnt = full_circ.count_ops().get('t', 0) + full_circ.count_ops().get('tdg', 0)
    
    final_u = Operator(full_circ).data
    d_final = aligned_dist(u_gate, final_u)
    
    print(f"    Hybrid T={t_cnt}, D={d_final}")
    return full_circ, t_cnt, d_final

qc_full = QuantumCircuit(2)
total_t = 0
total_ond_sq = 0

print("Synthesizing (Hybrid Beam-SK)...")

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
    
    if op.name == 'cx':
        qc_full.cx(qubits[0], qubits[1])
    elif op.name == 'u' or op.name == 'u3':
        theta, phi, lam = op.params
        qc_dummy = QuantumCircuit(1)
        qc_dummy.u(theta, phi, lam, 0)
        u_target = Operator(qc_dummy).data
        
        circ, t, d = solve_gate(u_target)
        total_t += t
        total_ond_sq += d**2
        
        for instr in circ.data:
            qc_full.append(instr.operation, [qubits[0]]) # Re-map

ond_final = aligned_dist(Operator(qc_full).data, U_target) # Approx OND of full
t_final = qc_full.count_ops().get('t', 0) + qc_full.count_ops().get('tdg', 0)

print(f"FINAL: T={t_final}, OND={ond_final}")

if ond_final < 0.1:
    print("SUCCESS")
    qiskit.qasm2.dump(qc_full, "./solution_challenge_10.qasm")
else:
    print("Optimization partial.")