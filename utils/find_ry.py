from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Operator
import numpy as np

ry = RYGate(np.pi/2).to_matrix()

qc = QuantumCircuit(1)
qc.ry(np.pi/2, 0)
qc_trans = transpile(qc, basis_gates=['u3', 'h', 's', 'sdg', 'z', 'x', 'y'], optimization_level=3)
print("Decomp:")
print(qc_trans.draw())

print("Trace Overlap:", np.trace(Operator(qc_trans).data.conj().T @ ry))

for instr in qc_trans.data:
    print(instr.operation.name)