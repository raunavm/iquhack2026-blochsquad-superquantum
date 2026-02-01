import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2


qc = QuantumCircuit(2)
qc.cx(0, 1)
qc.tdg(1)
qc.cx(0, 1)

print("Saving benchmark solution...")
qiskit.qasm2.dump(qc, "./solution_challenge_3_t1.qasm")