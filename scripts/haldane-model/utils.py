import numpy as np
from scipy.optimize import minimize
import warnings

from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Session, Estimator

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.hamiltonians import QuadraticHamiltonian
from qiskit_nature.second_q.circuit.library import SlaterDeterminant
from qiskit_nature.second_q.mappers import QubitMapper

from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms import IterativePhaseEstimation

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler
from qiskit_aer.noise import NoiseModel
from qiskit.providers import BackendV2

cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}
def band_structure_vqe_cost_func(params, ansatz, hamiltonian, estimator):
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]
 
    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
 
    return energy

def band_structure_exact(t1: float, t2: float, M: float, a_vecs: list[list[float]], b_vecs: list[list[float]], samples: int):
    x_list = np.linspace(-np.pi, np.pi, samples)
    y_list = np.linspace(-np.pi, np.pi, samples)
    result = {}
    for kx in x_list:
        for ky in y_list:
            k = [kx, ky]
            hx = hy = hz = 0
            for a in a_vecs:
                hx += t1*np.cos(np.dot(k, a))
                hy -= t1*np.sin(np.dot(k, a))
            hz += M
            for b in b_vecs:
                hz += 2*t2*np.sin(np.dot(k, b))
            result[(kx, ky)] = -np.sqrt(hx**2+hy**2+hz**2)
            print("E(["+str(round(kx, 3))+", "+str(round(ky, 3))+"]) = "+str(round(result[(kx, ky)], 3)))
    return result

def band_structure_vqe(t1: float, t2: float, M: float, a_vecs: list[list[float]], b_vecs: list[list[float]], samples: int, backend: BackendV2 = None) -> dict[list[float], float]:
    x_list = np.linspace(-np.pi, np.pi, samples)
    y_list = np.linspace(-np.pi, np.pi, samples)
    result = {}

    for kx in x_list:
        for ky in y_list:
            k = [kx, ky]
            hx = hy = hz = 0
            for a in a_vecs:
                hx += t1*np.cos(np.dot(k, a))
                hy -= t1*np.sin(np.dot(k, a))
            hz += M
            for b in b_vecs:
                hz += 2*t2*np.sin(np.dot(k, b))

            hamiltonian = SparsePauliOp(['X', 'Y', 'Z'], [hx, hy, hz])
            simulator = AerSimulator.from_backend(backend) if backend else AerSimulator()

            ansatz = efficient_su2(hamiltonian.num_qubits)
            pm = generate_preset_pass_manager(target=simulator.target, optimization_level=3)
            ansatz_isa = pm.run(ansatz)
            hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)
            x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)

            with Session(backend=simulator) as session:
                estimator = Estimator(mode=session)
                res = minimize(
                    band_structure_vqe_cost_func,
                    x0,
                    args=(ansatz_isa, hamiltonian_isa, estimator),
                    method="cobyla",
                )
            result[(kx, ky)] = float(res.fun)
            print("E(["+str(round(kx, 3))+", "+str(round(ky, 3))+"]) = "+str(round(result[(kx, ky)], 3)))

    return result

def real_space_slater_determinant(n_sites: int, t1: float, t2: float, phi: float, n_occ: int) -> SlaterDeterminant:
    lattice = [(i, (i + 1) % n_sites, 0) for i in range(n_sites)]+[(i, (i + 2) % n_sites, 1) for i in range(n_sites)]
    spin = 2
    H = 0.0 * np.zeros((n_sites*spin, n_sites*spin), dtype=complex)
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

    quadratic_hamiltonian = QuadraticHamiltonian(H)
    transformation_matrix, _, _ = quadratic_hamiltonian.diagonalizing_bogoliubov_transform()

    occupied_orbitals = transformation_matrix[:n_occ, :]
    return SlaterDeterminant(occupied_orbitals)

def real_space_fermionic_hamiltonian(n_sites: int, t1: float, t2: float, phi: float) -> FermionicOp:
    lattice = [(i, (i + 1) % n_sites, 0) for i in range(n_sites)]+[(i, (i + 2) % n_sites, 1) for i in range(n_sites)]
    spin = 2

    hamiltonian = 0.0 * FermionicOp({})
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
                    f"+_{s1} -_{s2}": t2*np.exp(1j*phi),
                    f"+_{s2} -_{s1}": t2*np.exp(-1j*phi)
                })
    
    return hamiltonian

def real_space_exact(n_sites: int, t1: float, t2: float, phi: float, n_occ: int) -> float:
    lattice = [(i, (i + 1) % n_sites, 0) for i in range(n_sites)]+[(i, (i + 2) % n_sites, 1) for i in range(n_sites)]
    spin = 2
    H = 0.0 * np.zeros((n_sites*spin, n_sites*spin), dtype=complex)
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

    eigvals, _ = np.linalg.eigh(H)
    result = np.sum(np.sort(eigvals)[:n_occ])
    print(f"Exact (n_sites={n_sites}, n_occ={n_occ}) = {result}")
    return result

def real_space_vqe(n_sites: int, t1: float, t2: float, phi: float, n_occ: int, mapper: QubitMapper, max_iters: int, backend: BackendV2 = None) -> float:
    fermionic_hamiltonian = real_space_fermionic_hamiltonian(n_sites, t1, t2, phi)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
    simulator = AerSimulator.from_backend(backend) if backend else AerSimulator()
    
    with Session(backend=simulator) as session:
        ansatz = real_space_slater_determinant(n_sites, t1, t2, phi, n_occ)
        ansatz_circuit = transpile(ansatz, backend=simulator, optimization_level=3)
    
        estimator = Estimator(mode=session)
        x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)
        
        cost_history_dict = {
            "prev_vector": None,
            "iters": 0,
            "cost_history": [],
        }
        def cost_func(params):
            if cost_history_dict["iters"] >= max_iters:
                return cost_history_dict["cost_history"][-1]
        
            pub = (ansatz_circuit, [qubit_hamiltonian], [params])
            result = estimator.run(pubs=[pub]).result()
            energy = result[0].data.evs[0]
        
            cost_history_dict["iters"] += 1
            cost_history_dict["prev_vector"] = params
            cost_history_dict["cost_history"].append(energy)
        
            return energy
        
        spsa = SPSA()
        res = spsa.minimize(cost_func, x0=x0)

    result = float(res.fun)
    print(f"VQE (n_sites={n_sites}, n_occ={n_occ}) = {result}")
    return result

def real_space_iqpe(n_sites: int, t1: float, t2: float, phi: float, n_occ: int, mapper: QubitMapper, t: float, n_trot: int, n_iters: int, backend: BackendV2 = None) -> float:
    fermionic_hamiltonian = real_space_fermionic_hamiltonian(n_sites, t1, t2, phi)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

    st = SuzukiTrotter(reps=n_trot)
    evolution = PauliEvolutionGate(qubit_hamiltonian, time=t)
    evolution_circuit = st.synthesize(evolution)
    initial_circuit = real_space_slater_determinant(n_sites, t1, t2, phi, n_occ)

    if backend:
        noise_model = NoiseModel.from_backend(backend) if backend else NoiseModel()
        sampler = Sampler(
            backend_options={
                "noise_model": noise_model,
                "basis_gates": noise_model.basis_gates,
            }
        )
    else:
        sampler = Sampler()

    # Sampler is deprecated but IQPE in Qiskit Algorithms has not been updated to use SamplerV2 yet 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iqpe = IterativePhaseEstimation(num_iterations=n_iters, sampler=sampler)
        
    res = iqpe.estimate(unitary=evolution_circuit, state_preparation=initial_circuit)

    result = float(-2*np.pi*res.phase/t)
    print(f"IQPE (n_sites={n_sites}, n_occ={n_occ}) = {result}")
    return result