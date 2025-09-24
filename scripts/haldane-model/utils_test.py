import numpy as np
from scipy.optimize import minimize
import warnings
import sys
import traceback
import threading

from qiskit import transpile, QuantumCircuit, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp

from qiskit_ibm_runtime import Session, Estimator

from qiskit_aer import AerSimulator
from qiskit.providers import BackendV2

from mthree import M3Mitigation
from qiskit.result import Counts
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import DynamicalDecoupling
from qiskit.circuit.library import XGate, YGate, SXGate, SXdgGate

USE_M3 = True
M3_SHOTS = 4096
USE_DD = True

_thread_local = threading.local()


def set_mitigation_config(use_m3=True, use_dd=True):
    global USE_M3, USE_DD
    USE_M3 = use_m3
    USE_DD = use_dd


def local_threading():
    if not hasattr(_thread_local, 'backend_objects'):
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2
        from qiskit_aer.noise import NoiseModel

        hw = FakeManilaV2()
        simulator = AerSimulator.from_backend(hw)
        noise_model = NoiseModel.from_backend(hw)
        simulator.set_options(noise_model=noise_model, basis_gates=noise_model.basis_gates)

        pm = generate_preset_pass_manager(target=simulator.target, optimization_level=3)

        _thread_local.backend_objects = {
            'hw': hw,
            'simulator': simulator,
            'noise_model': noise_model,
            'pass_manager': pm
        }
        _thread_local.m3_cache = {
            "backend_id": None,
            "mit": None,
            "qubits": None,
        }

    return _thread_local.backend_objects, _thread_local.m3_cache


def check_gpu_support() -> bool:
    try:
        sim = AerSimulator(device="GPU", method="statevector")
        qc = QuantumCircuit(1)
        qc.x(0)
        sim.run(qc, shots=1).result()
        return True
    except Exception:
        return False


GPU_AVAILABLE = check_gpu_support()


def band_structure_exact(kx: float, ky: float, t1: float, t2: float, M: float, a_vecs: list[list[float]],
                         b_vecs: list[list[float]]):
    k = [kx, ky]
    hx = hy = hz = 0
    for a in a_vecs:
        hx += t1 * np.cos(np.dot(k, a))
        hy -= t1 * np.sin(np.dot(k, a))
    hz += M
    for b in b_vecs:
        hz += 2 * t2 * np.sin(np.dot(k, b))
    result = -np.sqrt(hx ** 2 + hy ** 2 + hz ** 2)
    print(f"[EXACT] E([{round(kx, 3)}, {round(ky, 3)}]) = {result:.6f}")
    sys.stdout.flush()
    return result


def band_structure_vqe(kx: float, ky: float, t1: float, t2: float, M: float, a_vecs: list[list[float]],
                       b_vecs: list[list[float]], *,
                       exec_backend: BackendV2 = None,
                       schedule_backend: BackendV2 | None = None,
                       gpu_id: int | None = None) -> float:
    max_retries = 3
    for retry in range(max_retries):
        try:
            backend_objects, m3_cache = local_threading()
            if exec_backend is None:
                exec_backend = backend_objects['simulator']
            if schedule_backend is None:
                schedule_backend = backend_objects['hw']

            pm = backend_objects['pass_manager']

            mitigation_str = []
            if USE_M3: mitigation_str.append("M3")
            if USE_DD: mitigation_str.append("DD")
            mitigation_label = "+".join(mitigation_str) if mitigation_str else "RAW"

            sys.stdout.flush()

            k = [kx, ky]
            hx = hy = hz = 0
            for a in a_vecs:
                hx += t1 * np.cos(np.dot(k, a))
                hy -= t1 * np.sin(np.dot(k, a))
            hz += M
            for b in b_vecs:
                hz += 2 * t2 * np.sin(np.dot(k, b))

            print(f"  Hamiltonian constructed: hx={hx:.4f}, hy={hy:.4f}, hz={hz:.4f}")
            sys.stdout.flush()

            hamiltonian = SparsePauliOp(['X', 'Y', 'Z'], [hx, hy, hz])

            print(f"  Creating ansatz circuit...")
            sys.stdout.flush()
            ansatz = efficient_su2(1)

            ansatz_isa = pm.run(ansatz)
            hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)

            if USE_DD:
                ansatz_isa = _apply_dd(ansatz_isa, exec_backend, schedule_backend)

            x0 = 2 * np.pi * np.random.random(ansatz_isa.num_parameters)
            sys.stdout.flush()

            with Session(backend=exec_backend) as session:
                estimator = Estimator(mode=session)

                iteration_count = [0]

                def cost_func_with_progress(params):
                    try:
                        if USE_M3:
                            param_map = {p: v for p, v in zip(list(ansatz_isa.parameters), list(params))}
                            bound = ansatz_isa.assign_parameters(param_map, inplace=False)
                            result = _expectation_1q_m3(bound, hamiltonian, exec_backend, shots=M3_SHOTS,
                                                        m3_cache=m3_cache)
                        else:
                            result = _band_structure_vqe_cost_func(params, ansatz_isa, hamiltonian_isa, estimator)
                    except Exception as _m3_err:
                        result = _band_structure_vqe_cost_func(params, ansatz_isa, hamiltonian_isa, estimator)

                    iteration_count[0] += 1
                    if iteration_count[0] % 10 == 0:
                        print(f"    Iteration {iteration_count[0]}: E = {result:.6f}")
                        sys.stdout.flush()
                    return result

                res = minimize(
                    cost_func_with_progress,
                    x0,
                    args=(),
                    method="cobyla",
                    options={"maxiter": 100},
                )

            result = float(res.fun)
            device_info = f"GPU {gpu_id}" if (GPU_AVAILABLE and gpu_id is not None) else (
                f"CPU worker {gpu_id}" if gpu_id is not None else "CPU")
            print(f"[VQE DONE] E([{round(kx, 3)}, {round(ky, 3)}]) = {result:.6f} [{device_info}] [{mitigation_label}]")
            sys.stdout.flush()
            return result

        except RuntimeError as e:
            if "Already borrowed" in str(e) and retry < max_retries - 1:
                print(
                    f"[VQE RETRY] k-point [{round(kx, 3)}, {round(ky, 3)}]: Already borrowed error, retrying ({retry + 1}/{max_retries})")
                sys.stdout.flush()
                if hasattr(_thread_local, 'backend_objects'):
                    delattr(_thread_local, 'backend_objects')
                import time
                time.sleep(0.1 * retry)
                continue
            else:
                raise
        except Exception as e:
            print(f"[VQE ERROR] Failed at k-point [{round(kx, 3)}, {round(ky, 3)}]: {str(e)}")
            traceback.print_exc()
            sys.stdout.flush()
            raise


def _band_structure_vqe_cost_func(params, ansatz, hamiltonian, estimator):
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]
    return energy


def _apply_dd(circ_isa, exec_backend, schedule_backend=None):
    if schedule_backend is None:
        return circ_isa

    try:
        scheduled = transpile(
            circ_isa,
            backend=schedule_backend,
            optimization_level=0,
            scheduling_method="alap",
        )
    except Exception as e:
        return circ_isa

    try:
        durations = InstructionDurations.from_backend(schedule_backend)
    except Exception as e:
        return circ_isa

    def _has_duration(gate_name: str) -> bool:
        try:
            durations.get(gate_name, (0,))
            return True
        except Exception:
            return False

    if _has_duration("x") and _has_duration("y"):
        dd_sequence = [XGate(), YGate(), XGate(), YGate()]
        seq_name = "XY4"
    elif _has_duration("x"):
        dd_sequence = [XGate(), XGate()]
        seq_name = "XX-echo"
    elif _has_duration("sx"):
        dd_sequence = [SXGate(), SXdgGate(), SXGate(), SXdgGate()]
        seq_name = "SX/SXdg-echo"
    else:
        return circ_isa

    print(f"  [DD] Using sequence: {seq_name}")

    try:
        pm_dd = PassManager([DynamicalDecoupling(durations, dd_sequence)])
        dd_circ = pm_dd.run(scheduled)
    except Exception as e:
        print(f"[DD WARN] DD pass failed: {e}")
        return circ_isa

    try:
        retargeted = transpile(dd_circ, backend=exec_backend, optimization_level=0)
        return retargeted
    except Exception:
        return dd_circ


def _expectation_1q_m3(ansatz_isa, hamiltonian, simulator, shots=M3_SHOTS, m3_cache=None):
    if len(ansatz_isa.qubits) < 1:
        raise RuntimeError("Ansatz has no qubits")
    work_q = ansatz_isa.qubits[0]

    backend_id = id(simulator)
    qubits = (0,)

    if m3_cache is None:
        m3_cache = {"backend_id": None, "mit": None, "qubits": None}

    if m3_cache["mit"] is None or m3_cache["backend_id"] != backend_id or m3_cache["qubits"] != qubits:
        mit = M3Mitigation(simulator)
        mit.cals_from_system(qubits)
        m3_cache.update({"backend_id": backend_id, "mit": mit, "qubits": qubits})
    mit = m3_cache["mit"]

    circs, paulis, coeffs = [], [], []
    for label, coeff in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs):
        c = ansatz_isa.copy()

        if label == 'X':
            c.h(work_q)
        elif label == 'Y':
            c.sdg(work_q)
            c.h(work_q)
        elif label == 'Z':
            pass
        else:
            continue

        c.barrier(work_q)
        cr = ClassicalRegister(1, 'c')
        c.add_register(cr)
        c.measure(work_q, cr[0])

        if len(c.data) == 0:
            c.id(work_q)

        circs.append(c)
        paulis.append(label)
        coeffs.append(float(coeff.real))

    if len(circs) == 0:
        try:
            from qiskit_ibm_runtime import Session
            with Session(backend=simulator) as session:
                est = Estimator(mode=session)
                hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)
                pub = (ansatz_isa, [hamiltonian_isa], [np.zeros(ansatz_isa.num_parameters)])
                result = est.run(pubs=[pub], shots=shots).result()
                return float(result[0].data.evs[0])
        except Exception:
            raise RuntimeError("No Pauli terms generated and Estimator fallback failed")

    try:
        tcircs = transpile(circs, backend=simulator, optimization_level=0,
                           layout_method='trivial', routing_method='none', initial_layout=[0])
    except Exception:
        tcircs = transpile(circs, basis_gates=["rz", "sx", "x", "cx", "reset", "measure"],
                           optimization_level=0)

    job = simulator.run(tcircs, shots=shots)
    res = job.result()

    exp_terms = []
    for i, (label, coeff) in enumerate(zip(paulis, coeffs)):
        counts = res.get_counts(i)
        if not isinstance(counts, Counts):
            counts = Counts(counts)

        quasi = mit.apply_correction(counts, qubits)

        p0 = quasi.get('0', 0.0)
        p1 = quasi.get('1', 0.0)
        exp_z = p0 - p1
        exp_terms.append(coeff * exp_z)

    return float(sum(exp_terms))


_m3_cache = {}
_expectation_1q_m3 = lambda *args, **kwargs: _expectation_1q_m3(*args, **kwargs)