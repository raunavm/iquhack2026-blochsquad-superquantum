import numpy as np
from qiskit.quantum_info import Operator
import qiskit.qasm2
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

def aligned_ond(U, V):
    d = U
    u = V
    overlap = np.trace(d.conj().T @ u)
    if np.abs(overlap) < 1e-9: phase = 0
    else: phase = np.angle(overlap)
    return np.linalg.norm(d - u * np.exp(-1j * phase), ord=2)

U_target = 0.5 * np.array([
    [1, 1, 1, 1],
    [1, 1j, -1, -1j],
    [1, -1, 1, -1],
    [1, -1j, -1, 1j]
], dtype=complex)

print("Target Matrix:")
print(U_target)


qft = Operator(QFT(2, do_swaps=False)).data
print("\nQFT (no swaps):")
print(qft)

ond_qft = aligned_ond(U_target, qft)
print(f"QFT (no swaps) OND: {ond_qft}")

qft_swap = Operator(QFT(2, do_swaps=True)).data
print(f"QFT (with swaps) OND: {aligned_ond(U_target, qft_swap)}")

if ond_qft < 1e-9 or aligned_ond(U_target, qft_swap) < 1e-9:
    print("MATCH FOUND: It is QFT.")
    
    
    circuit = QuantumCircuit(2)
    
    
    
    
    qc = QuantumCircuit(2)
    qc.h(1) # MSB?
    qc.cp(np.pi/2, 0, 1)
    qc.h(0)
    
    import qiskit.quantum_info as qi
    if aligned_ond(U_target, qi.Operator(qc).data) < 1e-9:
        print("Circuit H(1) CP(0,1) H(0) matches.")
    else:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cp(np.pi/2, 1, 0)
        qc.h(1)
        if aligned_ond(U_target, qi.Operator(qc).data) < 1e-9:
             print("Circuit H(0) CP(1,0) H(1) matches.")
        else:
             print("Need to fix swaps.") 
             qc.swap(0, 1)
             if aligned_ond(U_target, qi.Operator(qc).data) < 1e-9:
                 print("Circuit matches with SWAP.")

    
    qiskit.qasm2.dump(qc, "./solution_challenge_8_exact.qasm")

else:
    print("Does not match standard QFT immediately.")