import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.synthesis import SolovayKitaev
from qiskit.quantum_info import Operator
import qiskit.qasm2

# Target: Ry(pi/14)
theta = np.pi / 14
qc_target = QuantumCircuit(1)
qc_target.ry(theta, 0)
u_target = Operator(qc_target).data

print(f"Target Ry({theta:.4f})")

# Try Degree 1 and 2
for deg in [1, 2, 3]:
    print(f"\n--- Solovay-Kitaev Degree {deg} ---")
    skd = SolovayKitaev(recursion_degree=deg)
    from qiskit.transpiler import PassManager
    pm = PassManager(skd)
    
    # Run
    qc_approx = pm.run(qc_target)
    
    # Flatten to basis
    qc_flat = transpile(qc_approx, basis_gates=['h','t','tdg','s','sdg','z','x','y','cx'], optimization_level=3)
    
    t_count = qc_flat.count_ops().get('t', 0) + qc_flat.count_ops().get('tdg', 0)
    
    # Calculate Dist
    op_approx = Operator(qc_flat).data
    d = u_target
    u = op_approx
    overlap = np.trace(d.conj().T @ u)
    phase = np.angle(overlap)
    dist = np.linalg.norm(d - u * np.exp(-1j * phase), ord=2)
    
    print(f"Single Qubit Result: T={t_count}, Dist={dist}")
    
    # Total Circuit Estimate
    # T_total = 2 * T_single
    # D_total approx 2 * D_single
    print(f"Estimated Total: T={2*t_count}, Dist={2*dist}")
    
    if 2*t_count < 20 and 2*dist < 0.027:
        print(">>> WINNER FOUND! <<<")
        # Construct Full
        full_qc = QuantumCircuit(2)
        # Ry(t/2) on q1
        # Need to re-synthesize Ry(pi/14) exactly?
        # qc_flat IS Ry(pi/14)?
        # Wait. CRyDecomp uses Ry(theta/2).
        # We targeted Ry(pi/14) which IS theta/2.
        # So qc_flat is the gate A.
        
        # Append A to q1
        for instr in qc_flat.data:
            full_qc.append(instr.operation, [1])
            
        full_qc.cx(0, 1)
        
        # Append Adag to q1
        # Inverse of qc_flat
        qc_inv = qc_flat.inverse()
        for instr in qc_inv.data:
            full_qc.append(instr.operation, [1])
            
        full_qc.cx(0, 1)
        
        # Flatten
        full_qc = transpile(full_qc, basis_gates=['h','t','tdg','s','sdg','z','x','cx'], optimization_level=3)
        
        qiskit.qasm2.dump(full_qc, "challenge_02/solution_challenge_2_opt_sk.qasm")
        break
