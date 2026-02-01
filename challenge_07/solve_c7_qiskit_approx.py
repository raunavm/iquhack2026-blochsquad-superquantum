from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_statevector
import qiskit.qasm2

target_sv = random_statevector(4, seed=42)


qc = QuantumCircuit(2)
qc.initialize(target_sv, [0, 1])

qc_exact = transpile(qc, basis_gates=['u', 'cx'], optimization_level=3)

from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit.transpiler import PassManager



sk = SolovayKitaev(recursion_degree=1) 

pm = PassManager(sk)

qc_approx = QuantumCircuit(2)
for instr in qc_exact.data:
    if instr.operation.name in ['u', 'u3']:
        sub_qc = QuantumCircuit(1)
        sub_qc.append(UnitaryGate(instr.operation.to_matrix()), [0])
        
        try:
            sub_approx = pm.run(sub_qc)
            
            q_idx = qc_exact.find_bit(instr.qubits[0]).index
            
            sub_trans = transpile(sub_approx, basis_gates=['h', 's', 'sdg', 't', 'tdg'])
            
            for sub_instr in sub_trans.data:
                qc_approx.append(sub_instr.operation, [qc_approx.qubits[q_idx]])
                
        except Exception as e:
            print(f"SK failed: {e}")
            qc_approx.append(instr.operation, instr.qubits)
            
    else:
        qc_approx.append(instr.operation, instr.qubits)

t_count = qc_approx.count_ops().get('t', 0) + qc_approx.count_ops().get('tdg', 0)
print(f"Approximated T Count: {t_count}")

qiskit.qasm2.dump(qc_approx, "/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_7_qiskit_approx.qasm")