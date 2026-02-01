import rmsynth
from qiskit import QuantumCircuit
import qiskit.qasm2
import numpy as np

qc = qiskit.qasm2.load("/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_7_sk_d1.qasm")
print(f"Original T Count: {qc.count_ops().get('t', 0) + qc.count_ops().get('tdg', 0)}")



print("Calling rmsynth via subprocess...")
import subprocess

infile = "/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_7_sk_d1.qasm"
outfile = "/Users/raunavmendiratta/Desktop/iQuHack/solution_challenge_7_rmsynth.qasm"

path = "/Users/raunavmendiratta/Library/Python/3.11/bin/rmsynth-optimize"

try:
    res = subprocess.run([path, infile, outfile], capture_output=True, text=True)
    
    if res.returncode != 0:
        print("rmsynth failed:")
        print(res.stderr)
    else:
        print("rmsynth success!")
        qc_opt = qiskit.qasm2.load(outfile)
        t_opt = qc_opt.count_ops().get('t', 0) + qc_opt.count_ops().get('tdg', 0)
        print(f"Optimized T Count: {t_opt}")

except Exception as e:
    print(f"Error: {e}")