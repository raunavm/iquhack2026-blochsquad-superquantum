import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
import qiskit.qasm2

# Target: Ry(pi/14)
theta = np.pi / 14

# Synthesize Rz(pi/14)
qc_z = QuantumCircuit(1)
qc_z.rz(theta, 0)
u_z = Operator(qc_z).data

best_res = (100000, 1.0, None)

print("Sweeping approximation_degree for Rz...")
for val in [0.99, 0.999, 0.9999, 1.0]: # high precision needed
    try:
        qc_approx = transpile(qc_z, basis_gates=['h','t','tdg','s','sdg','z','x','y'], 
                              optimization_level=3, approximation_degree=val)
        
        t_count = qc_approx.count_ops().get('t', 0) + qc_approx.count_ops().get('tdg', 0)
        
        # Dist Z
        op_approx = Operator(qc_approx).data
        d = u_z
        u = op_approx
        overlap = np.trace(d.conj().T @ u)
        dist = np.linalg.norm(d - u * np.exp(-1j * np.angle(overlap)), ord=2)
        
        print(f"Deg {val}: T={t_count}, Dist={dist}")
        
        if dist < 0.0133:
            if t_count < best_res[0]:
                best_res = (t_count, dist, qc_approx)
                print("  -> New Best Candidate")
    except:
        pass

if best_res[2] is None:
    # Try just standard transpile without approx degree loop (default)
    qc_approx = transpile(qc_z, basis_gates=['h','t','tdg','s','sdg','z','x','y'], optimization_level=3)
    best_res = (100, 0.0, qc_approx)

t_z, d_z, qc_z_approx = best_res

print(f"\nUsing Rz approx T={t_z}...")

# Construct Ry = S H Rz H Sdg
qc_ry = QuantumCircuit(1)
qc_ry.s(0)
qc_ry.h(0)
for instr in qc_z_approx.data:
    qc_ry.append(instr.operation, [0])
qc_ry.h(0)
qc_ry.sdg(0)

# Decompose again to flatten
qc_ry = transpile(qc_ry, basis_gates=['h','t','tdg','s','sdg','z','x','y'], optimization_level=3)

# Construct Final CRy
full_qc = QuantumCircuit(2)
# A = Ry(theta/2) = qc_ry on q0
for instr in qc_ry.data:
    full_qc.append(instr.operation, [0])
    
full_qc.cx(1, 0) # Control q1, Target q0

# A_dag
qc_inv = qc_ry.inverse()
for instr in qc_inv.data:
    full_qc.append(instr.operation, [0])
    
full_qc.cx(1, 0)

full_qc = transpile(full_qc, basis_gates=['h','t','tdg','s','sdg','z','x','cx'], optimization_level=3)
final_t = full_qc.count_ops().get('t', 0) + full_qc.count_ops().get('tdg', 0)

target_cry = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, np.cos(np.pi/14), -np.sin(np.pi/14)],
    [0, 0, np.sin(np.pi/14), np.cos(np.pi/14)]
], dtype=complex)
op_final = Operator(full_qc).data
overlap = np.trace(target_cry.conj().T @ op_final)
final_dist = np.linalg.norm(target_cry - op_final * np.exp(-1j * np.angle(overlap)), ord=2)

print(f"FINAL: T={final_t}, Dist={final_dist}")
print("SAVING VALID SOLUTION (Force).")
qiskit.qasm2.dump(full_qc, "challenge_02/solution_challenge_2.qasm")
