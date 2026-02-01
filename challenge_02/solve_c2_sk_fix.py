import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
import qiskit.qasm2

# Target: Ry(pi/14)
theta = np.pi / 14
qc_target = QuantumCircuit(1)
qc_target.ry(theta, 0)
u_target = Operator(qc_target).data

basis = ['h','t','tdg','s','sdg','z','x','y','cx']

# Try approximation_degree 1 to 5
# OR unitary_synthesis_method='sk'
print(f"Synthesizing Ry({theta:.4f}) with Qiskit SK...")

best_res = (10000, 1.0, None)

for deg in [1, 2, 3]: # approximation_degrees for 'sk' method not directly exposed?
    # Use generic transpile with SK
    # Note: 'sk' method usually requires 'transpiler.passes.synthesis.solovay_kitaev' available
    try:
        # We can pass method='sk' and plugin_config
        qc_approx = transpile(qc_target, basis_gates=basis, optimization_level=3, 
                              unitary_synthesis_method='sk', 
                              unitary_synthesis_plugin_config={'recursion_degree': deg})
        
        t_count = qc_approx.count_ops().get('t', 0) + qc_approx.count_ops().get('tdg', 0)
        
        op_approx = Operator(qc_approx).data
        # Dist
        d = u_target
        u = op_approx
        overlap = np.trace(d.conj().T @ u)
        phase = np.angle(overlap)
        dist = np.linalg.norm(d - u * np.exp(-1j * phase), ord=2)
        
        print(f"Degree {deg}: T={t_count}, Dist={dist}")
        
        if dist < 0.0133: # Valid single qubit
             if t_count < best_res[0]:
                 best_res = (t_count, dist, qc_approx)
                 
    except Exception as e:
        print(f"Degree {deg} failed: {e}")

if best_res[2] is not None:
    t_opt, d_opt, qc_opt = best_res
    print(f"\nConstructing Full Circuit with T={t_opt} (Total {2*t_opt})...")
    
    full_qc = QuantumCircuit(2)
    
    # Gate A
    for instr in qc_opt.data:
        full_qc.append(instr.operation, [1])
        
    full_qc.cx(0, 1)
    
    # Gate A_dag
    qc_inv = qc_opt.inverse()
    for instr in qc_inv.data:
        full_qc.append(instr.operation, [1])
        
    full_qc.cx(0, 1)
    
    full_qc = transpile(full_qc, basis_gates=basis, optimization_level=3)
    
    t_total = full_qc.count_ops().get('t', 0) + full_qc.count_ops().get('tdg', 0)
    
    # Dist
    target_cry = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.cos(np.pi/14), -np.sin(np.pi/14)],
        [0, 0, np.sin(np.pi/14), np.cos(np.pi/14)]
    ], dtype=complex)
    op_final = Operator(full_qc).data
    overlap = np.trace(target_cry.conj().T @ op_final)
    phase = np.angle(overlap)
    final_dist = np.linalg.norm(target_cry - op_final * np.exp(-1j * phase), ord=2)
    
    print(f"FINAL RESULT: T={t_total}, Dist={final_dist}")
    
    qiskit.qasm2.dump(full_qc, "challenge_02/solution_challenge_2_valid.qasm")
else:
    print("No valid SK solution found.")
