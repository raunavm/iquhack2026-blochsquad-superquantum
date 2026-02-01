import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2
from qiskit.quantum_info import Operator


def append_naive_cs(qc, c, t):
    qc.t(c)
    qc.t(t)
    qc.cx(c, t)
    qc.tdg(t)
    qc.cx(c, t)

def append_zero_controlled_naive_s(qc, c, t):
    qc.x(c) # Decompose X later
    append_naive_cs(qc, c, t)
    qc.x(c) # Decompose X later

def append_one_controlled_hz(qc, c, t):
    
    
    pass

qc = QuantumCircuit(2)

def x(q):
    qc.h(q); qc.s(q); qc.s(q); qc.h(q)

def z(q):
    qc.s(q); qc.s(q)

qc.h(1); qc.s(1); qc.s(1); qc.h(1) # X(1)
append_naive_cs(qc, 1, 0) # T=3
qc.h(1); qc.s(1); qc.s(1); qc.h(1) # X(1)


pass

pass


qiskit.qasm2.dump(qc, "./solution_c9_analysis.qasm")