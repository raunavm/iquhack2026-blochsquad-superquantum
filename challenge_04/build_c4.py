import qiskit.qasm2
from qiskit import QuantumCircuit, transpile

c3_t1 = qiskit.qasm2.load("./solution_challenge_3_t1.qasm")
c3_sk = qiskit.qasm2.load("./solution_challenge_3_sk_d2.qasm")

def append_circuit(qc, block, qubits):
    for instr in block.data:
        qc.append(instr.operation, [qubits[i] for i in [0,1]]) # Assuming 2 qubit blocks mapping 0->0, 1->1 directly

def build_c4(base_block, filename):
    qc = QuantumCircuit(2)
    qubits = qc.qubits
    
    
    
    qc.sdg(0); qc.sdg(1)
    qc.h(0); qc.h(1)
    
    
    for instr in base_block.data:
         mapped_qubits = []
         for q in instr.qubits:
             idx = base_block.qubits.index(q)
             mapped_qubits.append(qc.qubits[idx])
         qc.append(instr.operation, mapped_qubits)
         
    qc.h(0); qc.h(1)
    qc.s(0); qc.s(1)
    
    
    qc.h(0); qc.h(1)
    
    for instr in base_block.data:
         mapped_qubits = []
         for q in instr.qubits:
             idx = base_block.qubits.index(q)
             mapped_qubits.append(qc.qubits[idx])
         qc.append(instr.operation, mapped_qubits)
         
    qc.h(0); qc.h(1)
    
    qiskit.qasm2.dump(qc, filename)
    t_count = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)
    print(f"Saved {filename}: T={t_count}")

build_c4(c3_t1, "./solution_challenge_4_t2.qasm")

build_c4(c3_sk, "./solution_challenge_4_high_fidelity.qasm")