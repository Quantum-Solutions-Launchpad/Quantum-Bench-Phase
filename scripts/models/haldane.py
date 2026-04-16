import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
from qiskit.circuit.library import efficient_su2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, Estimator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.providers import BackendV2
from qiskit_algorithms.optimizers import SPSA
from loguru import logger


NAME = "haldane"
DISPLAY_NAME = "Haldane"
DEFAULT_PARAMS = {"t1": 1.0, "t2": 0.5, "phi": np.pi/4, "M": 0.0}
PARAM_LABELS = {"t1": "t_1", "t2": "t_2", "phi": "\\phi", "M": "M"}

def get_optimizer(max_iters):
    return SPSA(maxiter=max_iters)

def file_suffix(params):
    return f"t2-{params['t2']}"

def _build_H_matrix(n_sites, t1, t2, phi, M):
    lattice = [(i, (i + 1) % n_sites, 0) for i in range(n_sites)] + [(i, (i + 2) % n_sites, 1) for i in range(n_sites)]
    spin = 2
    H = np.zeros((n_sites * spin, n_sites * spin), dtype=complex)
    for i in range(n_sites):
        for s in range(spin):
            H[i * spin + s, i * spin + s] += M if i % 2 == 0 else -M
    for i, j, order in lattice:
        for s in range(spin):
            s1 = i * spin + s
            s2 = j * spin + s
            if order == 0:
                H[s1, s2] -= t1
                H[s2, s1] -= t1
            else:
                H[s1, s2] -= t2 * np.exp(1j * phi)
                H[s2, s1] -= t2 * np.exp(-1j * phi)
    return H

def fermionic_hamiltonian(n_sites, *, t1, t2, phi, M):
    lattice = [(i, (i + 1) % n_sites, 0) for i in range(n_sites)] + [(i, (i + 2) % n_sites, 1) for i in range(n_sites)]
    spin = 2

    hamiltonian = 0.0 * FermionicOp({})
    for i in range(n_sites):
        for s in range(spin):
            idx = i * spin + s
            hamiltonian += FermionicOp({f"+_{idx} -_{idx}": M if i % 2 == 0 else -M})
    for i, j, order in lattice:
        for s in range(spin):
            s1 = i * spin + s
            s2 = j * spin + s
            if order == 0:
                hamiltonian -= FermionicOp({
                    f"+_{s1} -_{s2}": t1,
                    f"+_{s2} -_{s1}": t1
                })
            else:
                hamiltonian -= FermionicOp({
                    f"+_{s1} -_{s2}": t2 * np.exp(1j * phi),
                    f"+_{s2} -_{s1}": t2 * np.exp(-1j * phi)
                })

    return hamiltonian

# Band structure (Haldane-only, used by haldane-model/band_structure_*.py)

LATTICE_VECTORS = {
    4: {
        "a_vecs": [np.array([0.0, -1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([-1.0, 0.0])],
        "b_vecs": [np.array([-1.0, -1.0]), np.array([1.0, -1.0]), np.array([-1.0, 1.0]), np.array([1.0, 1.0])],
    },
    6: {
        "a_vecs": [np.array([0.0, -1.0]), np.array([np.sqrt(3)/2, 0.5]), np.array([-np.sqrt(3)/2, 0.5])],
    },
}
# b_vecs for 6-site depend on a_vecs
_a6 = LATTICE_VECTORS[6]["a_vecs"]
LATTICE_VECTORS[6]["b_vecs"] = [_a6[1] - _a6[2], _a6[2] - _a6[0], _a6[0] - _a6[1]]

_cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}

def _band_structure_vqe_cost_func(params, ansatz, hamiltonian, estimator):
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    _cost_history_dict["iters"] += 1
    _cost_history_dict["prev_vector"] = params
    _cost_history_dict["cost_history"].append(energy)

    return energy

def band_structure_exact(kx, ky, t1, t2, M, a_vecs, b_vecs):
    k = [kx, ky]
    hx = hy = hz = 0
    for a in a_vecs:
        hx += t1 * np.cos(np.dot(k, a))
        hy -= t1 * np.sin(np.dot(k, a))
    hz += M
    for b in b_vecs:
        hz += 2 * t2 * np.sin(np.dot(k, b))

    result = -np.sqrt(hx**2 + hy**2 + hz**2)
    logger.info(f"E([{round(kx, 3)}, {round(ky, 3)}]) = {result}")
    return result

def band_structure_vqe(kx, ky, t1, t2, M, a_vecs, b_vecs, backend: BackendV2 = None):
    k = [kx, ky]
    hx = hy = hz = 0
    for a in a_vecs:
        hx += t1 * np.cos(np.dot(k, a))
        hy -= t1 * np.sin(np.dot(k, a))
    hz += M
    for b in b_vecs:
        hz += 2 * t2 * np.sin(np.dot(k, b))

    hamiltonian = SparsePauliOp(['X', 'Y', 'Z'], [hx, hy, hz])

    if backend:
        noise_model = NoiseModel.from_backend(backend) if backend else NoiseModel()
        simulator = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
    else:
        simulator = AerSimulator()

    ansatz = efficient_su2(hamiltonian.num_qubits)
    pm = generate_preset_pass_manager(target=simulator.target, optimization_level=3)
    ansatz_isa = pm.run(ansatz)
    hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)
    x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)

    with Session(backend=simulator) as session:
        estimator = Estimator(mode=session)
        res = minimize(
            _band_structure_vqe_cost_func,
            x0,
            args=(ansatz_isa, hamiltonian_isa, estimator),
            method="cobyla",
        )

    result = float(res.fun)
    logger.info(f"E([{round(kx, 3)}, {round(ky, 3)}]) = {result}")
    return result
