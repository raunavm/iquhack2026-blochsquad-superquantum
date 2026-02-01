import numpy as np
import qiskit.qasm2
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, transpile


phases_str = [
    "0",
    "1",
    "5/4",
    "7/4",
    "5/4",
    "7/4",
    "3/2",
    "3/2",
    "5/4",
    "7/4",
    "3/2",
    "3/2",
    "3/2",
    "3/2",
    "7/4",
    "5/4"
]

phases_frac = []
for s in phases_str:
    if s == "0": val = 0.0
    elif s == "1": val = 1.0
    elif "/" in s:
        n, d = map(int, s.split('/'))
        val = n/d
    phases_frac.append(val)
    
phases = np.array(phases_frac) * np.pi


def fast_walsh_hadamard(data):
    a = np.array(data, copy=True)
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a / n

spectrum = fast_walsh_hadamard(phases)

print("Spectrum (theta_k coeffs):")
qc = QuantumCircuit(4)

for k, coeff in enumerate(spectrum):
    if np.abs(coeff) < 1e-9: continue
    
    
    rot_angle = -2 * coeff
    
    
    frac = rot_angle / np.pi
    print(f"k={k:04b}, coeff={coeff/np.pi:.4f}pi, rot={frac:.4f}pi")
    
    qubits = [i for i in range(4) if (k >> i) & 1]
    
    if not qubits:
        continue
        
    
    if len(qubits) == 1:
        qc.rz(rot_angle, qubits[0])
    else:
        for i in range(len(qubits)-1):
            qc.cx(qubits[i], qubits[i+1])
        
        qc.rz(rot_angle, qubits[-1])
        
        for i in range(len(qubits)-2, -1, -1):
            qc.cx(qubits[i], qubits[i+1])

print("Synthesized.")
t_count = qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)
s_count = qc.count_ops().get('s', 0) + qc.count_ops().get('sdg', 0) # Rz(pi/2) is S
z_count = qc.count_ops().get('z', 0) # Rz(pi) is Z
rz_count = qc.count_ops().get('rz', 0)

print(f"Initial T-count (uncompiled Rz): {t_count} (Actual Rz: {rz_count})")

qc_trans = transpile(qc, basis_gates=['cx', 't', 'tdg', 's', 'sdg', 'z', 'h'], optimization_level=3)

ops = qc_trans.count_ops()
final_t = ops.get('t', 0) + ops.get('tdg', 0)
print(f"Final T-count: {final_t}")
print(f"Final ops: {ops}")

qiskit.qasm2.dump(qc_trans, "./solution_challenge_11.qasm")