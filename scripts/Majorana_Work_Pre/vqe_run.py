# pip install qiskit qiskit-aer qiskit-algorithms
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import SPSA

# ----- 1) Load scaffold -----
h_ij = np.loadtxt("out_vqe/vqe_terms_ij_ReIm.csv", delimiter=",", skiprows=1)  # i,j,Re,Im
pos  = np.loadtxt("out_vqe/vqe_positions_xy.csv", delimiter=",", skiprows=1)
N = pos.shape[0]

# Build the single-particle matrix h (Hermitian)
h = np.zeros((N, N), dtype=complex)
for i, j, Re, Im in h_ij:
    i, j = int(i), int(j)
    h[i, j] = Re + 1j*Im
# symmetrize just in case of tiny numeric asymmetries
h = (h + h.conj().T)/2

# Decide filling: number of negative eigenvalues at mu=0
evals = np.linalg.eigvalsh(h)
N_occ = int(np.sum(evals < 0.0))

# ----- 2) Build JW Pauli operator for H = sum_ij h_ij c_i^† c_j -----
def zstring(i, j):
    lo, hi = min(i, j), max(i, j)
    return ["I"]*(lo+1) + ["Z"]*(hi-lo-1) + ["I"]*(N-hi)

def term_paulis(i, j, Re, Im):
    # Number operator when i==j: Re * n_i = Re*(I - Z_i)/2
    if i == j:
        z = ["I"]*N; z[i] = "Z"
        return [(Re/2, SparsePauliOp.from_list([("".join(z), 1.0)])),
                (-Re/2, SparsePauliOp.from_list([("I"*N, 1.0)]))]
    # i != j: Re/2*(X_i Z... X_j + Y_i Z... Y_j) + Im/2*(X Z... Y - Y Z... X) * sgn
    sgn = 1 if i < j else -1
    baseZ = zstring(i, j)
    paulis = []
    # helper to place a single-qubit op at index k
    def place(op, k):
        s = ["I"]*N; s[k] = op; return s
    # Re terms
    s1 = place("X", i); s2 = place("X", j)
    p = ["".join([a if a!="I" else b for a,b in zip(s1, baseZ)])]
    q = ["".join([a if a!="I" else b for a,b in zip(s2, "I"*N)])]
    XX = "".join([p[0][k] if p[0][k]!="I" else q[0][k] for k in range(N)])
    s1 = place("Y", i); s2 = place("Y", j)
    p = ["".join([a if a!="I" else b for a,b in zip(s1, baseZ)])]
    q = ["".join([a if a!="I" else b for a,b in zip(s2, "I"*N)])]
    YY = "".join([p[0][k] if p[0][k]!="I" else q[0][k] for k in range(N)])
    paulis += [(0.5*Re, SparsePauliOp.from_list([(XX, 1.0)]))]
    paulis += [(0.5*Re, SparsePauliOp.from_list([(YY, 1.0)]))]
    # Im terms
    s1 = place("X", i); s2 = place("Y", j)
    XY = "".join([("".join([a if a!="I" else b for a,b in zip(s1, baseZ)]))[k]
                  if ("".join([a if a!="I" else b for a,b in zip(s1, baseZ)]))[k]!="I" else
                  ("".join([a if a!="I" else b for a,b in zip(s2, "I"*N)]))[k]
                  for k in range(N)])
    s1 = place("Y", i); s2 = place("X", j)
    YX = "".join([("".join([a if a!="I" else b for a,b in zip(s1, baseZ)]))[k]
                  if ("".join([a if a!="I" else b for a,b in zip(s1, baseZ)]))[k]!="I" else
                  ("".join([a if a!="I" else b for a,b in zip(s2, "I"*N)]))[k]
                  for k in range(N)])
    paulis += [(+0.5*Im*sgn, SparsePauliOp.from_list([(XY, 1.0)]))]
    paulis += [(-0.5*Im*sgn, SparsePauliOp.from_list([(YX, 1.0)]))]
    return paulis

H_op = SparsePauliOp.from_list([("I"*N, 0.0)])
for i in range(N):
    for j in range(N):
        Re, Im = np.real(h[i, j]), np.imag(h[i, j])
        if abs(Re) < 1e-12 and abs(Im) < 1e-12: 
            continue
        for coeff, op in term_paulis(i, j, Re, Im):
            H_op = (H_op + coeff * op).simplify()

# ----- 3) Add particle-number penalty: lambda*(N̂ - N_occ)^2 -----
# N̂ = sum_i (I - Z_i)/2  -> build once
lam = 10.0 * (np.max(evals) - np.min(evals) + 1e-6)  # safe big penalty
I = SparsePauliOp.from_list([("I"*N, 1.0)])
Nhat = SparsePauliOp.from_list([("I"*N, 0.0)])
for i in range(N):
    z = ["I"]*N; z[i] = "Z"
    Nhat += 0.5*SparsePauliOp.from_list([("".join(z), -1.0)]) + 0.5*I
Penalty = lam * (Nhat @ Nhat - 2*N_occ * Nhat + (N_occ**2) * I)
H_pen = (H_op + Penalty).simplify()

# ----- 4) VQE on simulator -----
algorithm_globals.random_seed = 7
ansatz = TwoLocal(N, rotation_blocks="ry", entanglement_blocks="cz", entanglement="linear", reps=2)
optimizer = SPSA(maxiter=150)

estimator = Estimator()  # statevector-like by default; swap to Aer Estimator for shots
vqe = VQE(estimator, ansatz, optimizer)
res = vqe.compute_minimum_eigenvalue(operator=H_pen)
print("VQE energy (penalized):", res.eigenvalue.real)

# (Optional) build an Estimator with QASM shots for a realistic run:
# from qiskit_aer.primitives import Estimator as AerEstimator
# estimator_shot = AerEstimator(shots=2000)
# vqe_shot = VQE(estimator_shot, ansatz, SPSA(maxiter=150))
# res_shot = vqe_shot.compute_minimum_eigenvalue(operator=H_pen)
# print("VQE(qasm) energy:", res_shot.eigenvalue.real)
