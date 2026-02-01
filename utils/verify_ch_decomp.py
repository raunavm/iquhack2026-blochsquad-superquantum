import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

def append_ry_pi4(qc, qubit):
    qc.sdg(qubit)
    qc.h(qubit)
    qc.t(qubit)
    qc.h(qubit)
    qc.s(qubit)

def append_ry_minus_pi4(qc, qubit):
    qc.sdg(qubit)
    qc.h(qubit)
    qc.tdg(qubit)
    qc.h(qubit)
    qc.s(qubit)

def append_cz(qc, c, t):
    qc.h(t)
    qc.cx(c, t)
    qc.h(t)

def append_ch(qc, c, t):
    append_ry_pi4(qc, t)
    qc.cx(c, t)
    append_ry_minus_pi4(qc, t)
    qc.cx(c, t)
    append_cz(qc, c, t)

qc_manual = QuantumCircuit(2)
append_ch(qc_manual, 1, 0)
op_manual = Operator(qc_manual).data

qc_lib = QuantumCircuit(2)
qc_lib.ch(1, 0)
op_lib = Operator(qc_lib).data

print("Manual CH vs Lib CH:")
print("Manual:\n", np.round(op_manual, 3))
print("Lib:\n", np.round(op_lib, 3))

diff = np.linalg.norm(op_manual - op_lib)
print(f"Diff: {diff}")

qc_cry = QuantumCircuit(2)
append_ry_pi4(qc_cry, 0)
qc_cry.cx(1, 0)
append_ry_minus_pi4(qc_cry, 0)
qc_cry.cx(1, 0)
op_cry = Operator(qc_cry).data

qc_ref = QuantumCircuit(2)
qc_ref.cry(np.pi/2, 1, 0)
op_ref = Operator(qc_ref).data

print("Manual CRy vs Lib CRy:")
print("Diff:", np.linalg.norm(op_cry - op_ref))
print("Manual CRy:\n", np.round(op_cry, 3))
print("Lib CRy:\n", np.round(op_ref, 3))