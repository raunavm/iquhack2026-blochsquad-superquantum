# iQuHack 2026 - Superquantum Challenge Solutions

**Team**: Blochsquad
**Challenge**: Superquantum (Clifford+T Compilation)

This repository contains optimized quantum circuit compilations for the 11 challenges presented at iQuHack 2026. Our solutions focus on minimizing T-gate counts and Operator Norm Distance (OND).

## Structure

The repository is organized by challenge number:

- **`challenge_01` - `challenge_11`**: Contains the source code (Python solvers) and the final QASM solution files for each challenge.
- **`utils`**: Helper scripts for analysis, decomposition, and verification.

## Key Results

| Challenge | Goal | T-Count | OND | Status |
|---|---|---|---|---|
| **#1** | Controlled-Y | 0 | 0.0 | Exact |
| **#6** | Transverse Ising | 3 | 0.3169 | Pareto Optimal |
| **#9** | Structured Unitary 2 | 3 | 0.0 | Exact |
| **#10** | Random Unitary (4x4) | 6,543 | 0.023 | Valid (<0.1 OND) |
| **#11** | Diagonal Unitary | 11 | 0.0 | Exact |

## Usage

Each folder contains a `solve_*.py` script that can be run to regenerate the `.qasm` solution files.

```bash
python3 challenge_11/solve_c11_walsh.py
```

## License

MIT License
