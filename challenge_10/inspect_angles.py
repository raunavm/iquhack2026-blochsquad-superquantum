import numpy as np
from qiskit.quantum_info import Operator, random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.circuit.library import CXGate

seed = 42
U_target = random_unitary(4, seed=seed).data
kak_decomp = TwoQubitBasisDecomposer(gate=CXGate(), euler_basis='U3')
qc_kak = kak_decomp(U_target)

print("KAK ops:", qc_kak.count_ops())

for instr in qc_kak.data:
    if instr.operation.name == 'u3':
        t, p, l = instr.operation.params
        print(f"U3: theta={t:.4f}, phi={p:.4f}, lam={l:.4f}")
        for val, name in [(t, 'theta'), (p, 'phi'), (l, 'lam')]:
            frac = val / np.pi
            print(f"  {name}/pi = {frac:.4f}")