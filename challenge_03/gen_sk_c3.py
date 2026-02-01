import numpy as np
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit, transpile
import qiskit.qasm2

target_angle = - 2 * np.pi / 7
target_rz = np.diag([np.exp(-1j * target_angle/2), np.exp(1j * target_angle/2)])

print(f"Goal: Beat OND 0.05609")

for depth in [1, 2]:
    print(f"\n--- SK Depth {depth} ---")
    sk = SolovayKitaev(recursion_degree=depth)
    qc = QuantumCircuit(1)
    qc.append(UnitaryGate(target_rz), [0])
    
    from qiskit.transpiler import PassManager
    pm = PassManager(sk)
    qc_k = pm.run(qc)
    qc_trans = transpile(qc_k, basis_gates=['h', 's', 'sdg', 't', 'tdg'])
    
    full = QuantumCircuit(2)
    full.cx(0, 1)
    for instr in qc_trans.data:
        full.append(instr.operation, [full.qubits[1]]) # Apply on q1
    full.cx(0, 1)
    
    t_count = full.count_ops().get('t', 0) + full.count_ops().get('tdg', 0)
    print(f"T Count: {t_count}")
    
    
    qiskit.qasm2.dump(full, f"./solution_challenge_3_sk_d{depth}.qasm")