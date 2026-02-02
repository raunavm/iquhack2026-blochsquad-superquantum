import subprocess
import numpy as np
import os
import sys

# Try importing PyZX (Critical for lowering T count)
try:
    import pyzx as zx
    HAS_PYZX = True
except ImportError:
    HAS_PYZX = False

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
# Disabled library import to force binary usage (Lib was producing 17k gates)
HAS_GRIDSYNTH_LIB = False
import qiskit.qasm2

# --- SETTINGS ---
ROTATION_ANGLE = -2 * np.pi / 7
EPSILON = 2.0e-6 # Default, overridden by sweep
GRIDSYNTH_PATH = "./gridsynth_mac"
OUTPUT_FILE = "solution_challenge_4.qasm"

def clean_angle(angle):
    return angle % (2 * np.pi)

def synthesize_angle_robust(angle, epsilon):
    """Synthesizes Rz(angle) using gridsynth binary."""
    angle = clean_angle(angle)
    if abs(angle) < 1e-9: return QuantumCircuit(1)
    
    # Handle exact Clifford angles
    steps = angle / (np.pi / 4)
    if abs(steps - round(steps)) < 1e-9:
        k = int(round(steps)) % 8
        qc = QuantumCircuit(1)
        if k == 1: qc.t(0)
        elif k == 2: qc.s(0)
        elif k == 3: qc.s(0); qc.t(0)
        elif k == 4: qc.s(0); qc.s(0) # Z
        elif k == 5: qc.s(0); qc.s(0); qc.t(0)
        elif k == 6: qc.sdg(0)
        elif k == 7: qc.tdg(0)
        return qc

    if HAS_GRIDSYNTH_LIB:
        try:
            # Gridsynth library usage
            return gridsynth_rz(angle, epsilon)
        except Exception as e:
            pass # Fallback

    if not os.access(GRIDSYNTH_PATH, os.X_OK):
        os.chmod(GRIDSYNTH_PATH, 0o755)
    
    try:
        # Use clean angle to avoid negative flag issues
        cmd = [GRIDSYNTH_PATH, str(angle), "-e", str(epsilon)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0: 
            return None
        
        seq = ""
        # DEBUG: Print output
        print(f"DEBUG: Gridsynth output length: {len(result.stdout)}")
        # print(f"DEBUG: Output snippet: {result.stdout[:200]}...")
        
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line and all(c in "HTSXYZWabcdef" for c in line) and len(line) > 0:
                seq = line
                # DEBUG: Found sequence
                print(f"DEBUG: Parsed sequence length: {len(seq)}")
                break
        if not seq: 
            print("DEBUG: No valid sequence found.")
            return QuantumCircuit(1)

        qc = QuantumCircuit(1)
        for c in seq:
            if c == 'H': qc.h(0)
            elif c == 'T': qc.t(0)
            elif c == 't': qc.tdg(0)
            elif c == 'S': qc.s(0)
            elif c == 's': qc.sdg(0)
            elif c == 'X': qc.h(0); qc.s(0); qc.s(0); qc.h(0) 
            elif c == 'Y': qc.sdg(0); qc.h(0); qc.sdg(0)
            elif c == 'Z': qc.s(0); qc.s(0)
            elif c == 'W': qc.s(0); qc.h(0)
        return qc
    except Exception as e:
        return None

def optimize_and_measure(full_qc):
    """Optimizes logic (limited) and measures OND/T-count."""
    # PyZX disabled due to corruption issues (OND -> 2.0)
    final_qc = full_qc
    
    ops_pre = full_qc.count_ops()
    t_pre = ops_pre.get('t', 0) + ops_pre.get('tdg', 0)
    print(f"  DEBUG: Pre-transpile Ops: {ops_pre} (T={t_pre})")
    
    # SKIP TRANSPILATION (It explodes T-count for unknown reasons)
    final_qc = final_qc
    
    # Check Metrics
    ops = final_qc.count_ops()
    t_count = ops.get('t', 0) + ops.get('tdg', 0)
    print(f"  DEBUG: Final Ops: {ops} (T={t_count})")
    
    # Calculate Dist
    XX = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], dtype=complex)
    YY = np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]], dtype=complex)
    
    from scipy.linalg import expm
    U_exact = expm(1j * np.pi/7 * (XX + YY))
    
    U_approx = Operator(final_qc).data
    d = np.trace(U_exact.conj().T @ U_approx)
    phase = d / abs(d) if abs(d) > 1e-9 else 1
    ond = np.linalg.norm(U_exact - (phase * U_approx), 2)
    
    return t_count, ond, final_qc

def main():
    print(f"--- SOLVING CHALLENGE 4: HAMILTONIAN SIM (SWEEP) ---")
    print(f"Angle: {ROTATION_ANGLE}")
    
    EPSILONS = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    best_qc = None
    best_t = float('inf')
    best_ond = float('inf')
    
    print(f"{'Epsilon':<10} | {'T':<8} | {'OND':<12} | {'Status'}")
    print("-" * 50)
    
    for eps in EPSILONS:
        # 1. Synthesize core
        rz_core = synthesize_angle_robust(ROTATION_ANGLE, eps)
        if not rz_core: continue
        
        # 2. Build XX
        qc_xx = QuantumCircuit(2)
        qc_xx.h(0); qc_xx.h(1)
        qc_xx.cx(0, 1)
        qc_xx.compose(rz_core, [1], inplace=True)
        qc_xx.cx(0, 1)
        qc_xx.h(0); qc_xx.h(1)
        
        # 3. Build YY
        qc_yy = QuantumCircuit(2)
        qc_yy.sdg(0); qc_yy.sdg(1)
        qc_yy.h(0); qc_yy.h(1)
        qc_yy.cx(0, 1)
        qc_yy.compose(rz_core, [1], inplace=True)
        qc_yy.cx(0, 1)
        qc_yy.h(0); qc_yy.h(1)
        qc_yy.s(0); qc_yy.s(1)
        
        full_qc = qc_xx.compose(qc_yy)
        
        # Optimize & Measure
        t, ond, qc_opt = optimize_and_measure(full_qc)
        
        status = "FAIL"
        if ond < 1e-4: # Acceptance threshold
            status = "PASS"
            if t < best_t:
                status = "BEST"
                best_t = t
                best_ond = ond
                best_qc = qc_opt
        
        print(f"{eps:.1e}    | {t:<8} | {ond:.2e}     | {status}")

    print("-" * 50)
    
    if best_qc:
        print(f"VICTORY: T={best_t}, OND={best_ond:.2e}")
        with open(OUTPUT_FILE, "w") as f:
            qiskit.qasm2.dump(best_qc, f)
    else:
        print("FAILURE: No solution met criteria.")

if __name__ == "__main__":
    main()
