import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Operator

ry_target = RYGate(np.pi/4).to_matrix()
print("Target Ry(pi/4):")
print(np.round(ry_target, 3))

S = np.array([[1,0],[0,1j]])
H = 1/np.sqrt(2)*np.array([[1,1],[1,-1]])
T = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])
Sdg = np.array([[1,0],[0,-1j]])
op = S @ H @ T @ H @ Sdg
print("Constructed Op (S H T H Sdg):")
print(np.round(op, 3))

def overlap(U, V):
    tr = np.trace(U.conj().T @ V)
    return np.abs(tr) / 2

print(f"Overlap: {overlap(ry_target, op)}")

ratio = op[0,0] / ry_target[0,0]
print(f"Phase Ratio: {ratio}, Angle: {np.angle(ratio)}")