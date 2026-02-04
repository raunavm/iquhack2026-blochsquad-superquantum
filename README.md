# iQuHack 2026 - Superquantum Challenge

**Team**: Blochsquad  
**Challenge**: Clifford+T Circuit Compilation

## Overview

Solutions to all 12 Superquantum challenges, synthesizing quantum circuits using the Clifford+T gate set {H, S, T, CNOT} while minimizing T-gate count and operator norm distance (OND).

📄 **[Full Technical Report (PDF)](specifications/MIT_iQuHack_Superquantum.pdf)** — Detailed writeup with mathematical derivations and implementation details.

## Results

| Challenge | Description | T-count | OND | Method |
|-----------|-------------|---------|-----|--------|
| 1 | Controlled-Y | **0** | 0.0 | Clifford identity |
| 2 | C-Ry(π/7) | **0** | 0.224 | Vectorized search |
| 3 | exp(iπ/7 ZZ) | **1** | 0.056 | Priority queue search |
| 4 | exp(iπ/7(XX+YY)) | **2** | <10⁻⁴ | Gridsynth |
| 5 | exp(iπ/4(XX+YY+ZZ)) | **0** | 0.0 | SWAP identity |
| 6 | Ising model | **0** | 0.0 | Clifford analysis |
| 7 | Random state | **0** | N/A | Clifford search |
| 8 | Structured U1 | **3** | 0.0 | CS decomposition |
| 9 | Structured U2 | **3** | ≈0 | Algebraic synthesis |
| 10 | Random U(4) | **4364** | <0.1 | KAK + Solovay-Kitaev |
| 11 | 4-qubit diagonal | **11** | 0.0 | Walsh-Hadamard |
| 12 | Commuting Paulis | **1** | ≈0 | Stabilizer diagonalization |

## Repository Structure

```
├── challenge_01/ - challenge_12/    # Solutions for each challenge
│   ├── solution_challenge_*.qasm    # Final QASM output
│   └── solve_*.py                   # Python solver script
├── specifications/                   # Challenge PDFs and technical report
│   ├── MIT_iQuHack_Superquantum.pdf # Full technical writeup
│   ├── challenges1-11.pdf           # Problem specifications
│   └── challenge12.pdf              # Bonus challenge spec
└── README.md
```

## Usage

Each challenge folder contains a solver script that generates the `.qasm` solution:

```bash
python3 challenge_11/solve_c11_walsh.py
```

## Key Methods

- **Algebraic Decomposition**: CY, SWAP, structured unitaries (Challenges 1, 5, 8, 9)
- **Walsh-Hadamard Transform**: Diagonal unitary synthesis (Challenge 11)
- **Stabilizer Diagonalization**: Commuting Pauli Hamiltonians (Challenge 12)
- **Gridsynth**: Optimal Rz approximation (Challenges 3, 4)
- **Solovay-Kitaev**: Universal single-qubit synthesis (Challenge 10)

## License

MIT License
