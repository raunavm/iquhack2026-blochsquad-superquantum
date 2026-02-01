import numpy as np
from qiskit.quantum_info import random_statevector, Statevector
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit.transpiler import PassManager
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.circuit.library import UnitaryGate
import qiskit.qasm2

target_sv = random_statevector(4, seed=42)
print("Target State Data:")
print(target_sv.data)


qc = QuantumCircuit(2)
qc.initialize(target_sv, [0, 1])
qc_trans = transpile(qc, basis_gates=['u', 'cx'], optimization_level=3)
print("\nExact Circuit (U/CX):")
print(qc_trans.draw())


state_overlap = lambda sv1, sv2: np.abs(sv1.inner(sv2))**2

for depth in [1, 2]:
    print(f"\n--- SK Depth {depth} ---")
    sk = SolovayKitaev(recursion_degree=depth)
    pm = PassManager(sk)
    
    qc_sk_input = QuantumCircuit(2)
    for instr in qc_trans.data:
        if instr.operation.name in ['u', 'u3']:
             qc_sk_input.append(UnitaryGate(instr.operation.to_matrix()), [qc_sk_input.qubits[qc_trans.find_bit(q).index] for q in instr.qubits])
        else:
             qc_sk_input.append(instr.operation, [qc_sk_input.qubits[qc_trans.find_bit(q).index] for q in instr.qubits])

    qc_approx = pm.run(qc_sk_input)
    qc_basis = transpile(qc_approx, basis_gates=['h', 's', 'sdg', 't', 'tdg', 'cx'])
    
    t_count = qc_basis.count_ops().get('t', 0) + qc_basis.count_ops().get('tdg', 0)
    
    final_sv = Statevector.from_instruction(qc_basis)
    fid = state_overlap(target_sv, final_sv)
    
    print(f"T Count: {t_count}")
    print(f"State Fidelity: {fid}")
    print(f"State Infidelity: {1-fid}")
    
    filename = f"/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_7_sk_d{depth}.qasm"
    qiskit.qasm2.dump(qc_basis, filename)

best_fid_t0 = 0
best_qc_t0 = QuantumCircuit(2)
from qiskit.quantum_info import random_clifford

print("\nSearching T=0 (Clifford) best fit...")
for i in range(5000):
   c = random_clifford(2)
   sv = Statevector.from_instruction(c)
   fid = state_overlap(target_sv, sv)
   if fid > best_fid_t0:
       best_fid_t0 = fid
       best_qc_t0 = c.to_circuit()

print(f"Best T=0 Fidelity: {best_fid_t0}")
qc_t0_trans = transpile(best_qc_t0, basis_gates=['h', 's', 'sdg', 'cx'])
qiskit.qasm2.dump(qc_t0_trans, "/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_7_t0.qasm")