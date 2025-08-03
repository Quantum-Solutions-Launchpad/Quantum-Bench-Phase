import numpy as np
from scipy.optimize import minimize

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Session, Estimator

from qiskit_aer import AerSimulator
from qiskit.providers import BackendV2

cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}
def vqe_cost_func(params, ansatz, hamiltonian, estimator):
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]
 
    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
 
    return energy

def haldane_momentum_vqe(t1: float, t2: float, M: float, a_vecs: list[list[float]], b_vecs: list[list[float]], samples: int, backend: BackendV2 = None) -> dict[list[float], float]:
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
                    vqe_cost_func,
                    x0,
                    args=(ansatz_isa, hamiltonian_isa, estimator),
                    method="cobyla",
                )
            result[(kx, ky)] = float(res.fun)
            print("E(["+str(round(kx, 3))+", "+str(round(ky, 3))+"]) = "+str(round(res.fun, 3)))

    return result