import re
import glob

def check_qubits(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    match = re.search(r'qreg\s+(\w+)\[(\d+)\];', content)
    if not match:
        print(f"{filename}: No qreg found!")
        return
    
    name, size = match.groups()
    size = int(size)
    print(f"{filename}: Declared {size} qubits ({name})")
    
    max_idx = -1
    for m in re.finditer(rf'{name}\[(\d+)\]', content):
        idx = int(m.group(1))
        if idx > max_idx: max_idx = idx
    
    print(f"  Max index used: {max_idx}")
    if max_idx >= size:
        print("  ERROR: Index out of bounds!")

files = glob.glob("solution_challenge_*.qasm")
files.sort()
for f in files:
    check_qubits(f)