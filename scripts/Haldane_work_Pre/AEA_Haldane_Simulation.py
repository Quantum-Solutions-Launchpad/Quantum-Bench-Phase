import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from pennylane.qchem.utils import sparse_hamiltonian

# Parameters
n_cells = [2]
h1 = 0.5
h2 = 1.0
phi = 0.1

# Build qubit Hamiltonian using PennyLane's spin.haldane
H = qml.spin.haldane(
    lattice="chain",
    n_cells=n_cells,
    hopping=h1,
    hopping_next=h2,
    phi=phi,
    mapping="jordan_wigner",
    boundary_condition=False,
)

# Convert to matrix and diagonalize
H_matrix = sparse_hamiltonian(H).toarray()
eigvals, _ = eigh(H_matrix)

# Plot eigenvalues
plt.plot(eigvals, 'o')
plt.title("PennyLane Haldane Model (Chain, 4 Qubits)")
plt.xlabel("Eigenstate Index")
plt.ylabel("Energy")
plt.grid(True)
plt.tight_layout()
plt.show()
