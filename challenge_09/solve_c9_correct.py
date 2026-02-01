import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2
from qiskit.quantum_info import Operator

def aligned_ond(U, V):
    d = U
    u = V
    overlap = np.trace(d.conj().T @ u)
    if np.abs(overlap) < 1e-9: phase = 0
    else: phase = np.angle(overlap)
    return np.linalg.norm(d - u * np.exp(-1j * phase), ord=2)

a = (1+1j)/2
M_target = np.array([
    [1, 0, 0, 0],
    [0, 0, -a, a],
    [0, 1j, 0, 0],
    [0, 0, -a, -a]
], dtype=complex)

qc = QuantumCircuit(2)

qc.x(1)
qc.t(1); qc.t(0); qc.cx(1, 0); qc.tdg(0); qc.cx(1, 0)
qc.x(1)


qc.t(1); qc.t(0); qc.cx(1, 0); qc.tdg(0); qc.cx(1, 0)
qc.ch(1, 0)
qc.t(1); qc.t(0); qc.cx(1, 0); qc.tdg(0); qc.cx(1, 0)
qc.ch(1, 0)
qc.cz(1, 0)

qc.z(1)
qc.t(1)

qc.cx(0, 1); qc.cx(1, 0); qc.cx(0, 1)

ond = aligned_ond(M_target, Operator(qc).data)
print(f"Initial OND (with Z T): {ond}")

if ond > 0.1:
    phases = [
        ([], "None"),
        ([("z",1)], "Z"),
        ([("t",1)], "T"),
        ([("tdg",1)], "Tdg"),
        ([("z",1), ("t",1)], "Z T"),
        ([("z",1), ("tdg",1)], "Z Tdg"),
        ([("s",1)], "S"),
        ([("sdg",1)], "Sdg")
    ]
    
    best_ond = 100
    best_ops = []
    
    for ops, name in phases:
        qc_test = QuantumCircuit(2)
        qc_test.x(1)
        qc_test.t(1); qc_test.t(0); qc_test.cx(1, 0); qc_test.tdg(0); qc_test.cx(1, 0)
        qc_test.x(1)
        qc_test.t(1); qc_test.t(0); qc_test.cx(1, 0); qc_test.tdg(0); qc_test.cx(1, 0)
        qc_test.ch(1, 0)
        qc_test.t(1); qc_test.t(0); qc_test.cx(1, 0); qc_test.tdg(0); qc_test.cx(1, 0)
        qc_test.ch(1, 0)
        qc_test.cz(1, 0)
        
        for op, qubit in ops:
            getattr(qc_test, op)(qubit)
            
        qc_test.cx(0, 1); qc_test.cx(1, 0); qc_test.cx(0, 1)
        
        curr_ond = aligned_ond(M_target, Operator(qc_test).data)
        print(f"Phase {name}: OND={curr_ond}")
        
        if curr_ond < best_ond:
            best_ond = curr_ond
            best_ops = ops

    print(f"Best Phase sequence: {best_ops} with OND {best_ond}")
    
    qc_final = QuantumCircuit(2)
    qc_final.x(1)
    qc_final.t(1); qc_final.t(0); qc_final.cx(1, 0); qc_final.tdg(0); qc_final.cx(1, 0)
    qc_final.x(1)
    qc_final.t(1); qc_final.t(0); qc_final.cx(1, 0); qc_final.tdg(0); qc_final.cx(1, 0)
    qc_final.ch(1, 0)
    qc_final.t(1); qc_final.t(0); qc_final.cx(1, 0); qc_final.tdg(0); qc_final.cx(1, 0)
    qc_final.ch(1, 0)
    qc_final.cz(1, 0)
    
    for op, qubit in best_ops:
        getattr(qc_final, op)(qubit)
        
    qc_final.cx(0, 1); qc_final.cx(1, 0); qc_final.cx(0, 1)
    
    t_count = qc_final.count_ops().get('t', 0) + qc_final.count_ops().get('tdg', 0)
    print(f"Final Count T={t_count}")
    qiskit.qasm2.dump(qc_final, "/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_9.qasm")