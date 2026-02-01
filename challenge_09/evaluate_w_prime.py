import numpy as np

b = (-1+1j)/2
c = (1+1j)/2
d = (-1-1j)/2

M_block1 = np.array([[b, c], [b, d]], dtype=complex)
Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)

W_prime = Sdg @ M_block1
print("W_prime:")
print(np.round(W_prime, 3))

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]])

gates = {'I':I, 'X':X, 'Y':Y, 'Z':Z, 'H':H, 'S':S, 'Sdg':Sdg}

def check_match(U, name, G):
    tr = np.trace(U.conj().T @ G)
    dist = 1 - np.abs(tr)/2.0
    if dist < 1e-4:
        print(f"Match found! {name}")
        phase = np.angle(tr)
        print(f"Phase: {np.exp(1j*phase)}")
        return True
    return False

found = False
for name, G in gates.items():
    if check_match(W_prime, name, G):
        found = True

if not found:
    print("Checking Combinations...")
    HZ = H @ Z
    check_match(W_prime, "HZ", HZ)
    
    HX = H @ X
    check_match(W_prime, "HX", HX)

    HS = H @ S
    check_match(W_prime, "HS", HS)