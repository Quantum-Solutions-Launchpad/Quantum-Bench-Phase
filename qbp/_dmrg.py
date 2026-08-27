"""ITensorMPS DMRG simulation method (Julia bridge)."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from qbp._method import Method, ParamSpec, SimulationMethod, register_method
from qbp._model import Model


def _complex_payload(value):
    c = complex(value)
    return {"re": c.real, "im": c.imag}


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _fermionic_terms(op):
    return [
        {"label": label, "coefficient": _complex_payload(coeff)}
        for label, coeff in op.items()
    ]


def _pauli_terms(pauli_op, *, scale: float = 1.0) -> list[dict]:
    simplified = pauli_op.simplify()
    return [
        {"pauli": str(pauli), "coefficient": _complex_payload(scale * coeff)}
        for pauli, coeff in zip(simplified.paulis, simplified.coeffs)
        if abs(coeff) > 1e-14 and set(str(pauli)) != {"I"}
    ]


def _pauli_constant_shift(pauli_op, *, scale: float = 1.0) -> complex:
    simplified = pauli_op.simplify()
    shift = 0.0j
    for pauli, coeff in zip(simplified.paulis, simplified.coeffs):
        if abs(coeff) > 1e-14 and set(str(pauli)) == {"I"}:
            shift += scale * complex(coeff)
    return shift


def _export_pauli_operator(
    pauli_op,
    path: str,
    observable: str = "E",
    *,
    scale: float = 1.0,
    label: str = "hamlib_operator",
):
    """Export a SparsePauliOp as native Pauli terms for the Julia DMRG bridge."""
    n_qubits = int(pauli_op.num_qubits)

    payload = {
        "format": "qbp_pauli_op_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "hamlib",
        "display_name": label,
        "lattice": None,
        "spin": "pauli",
        "n_sites": n_qubits,
        "n_qubits": n_qubits,
        "n_occ": 0,
        "model_params": {},
        "terms": _pauli_terms(pauli_op, scale=scale),
        "constant_shift": _complex_payload(_pauli_constant_shift(pauli_op, scale=scale)),
        "observable": observable,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def _pauli_initial_state(initial_state: str) -> str:
    return "neel" if initial_state == "packed" else initial_state


def _pauli_exact_energy(pauli_op, *, extremum: str = "min") -> float:
    matrix = pauli_op.to_matrix()
    eigvals = np.linalg.eigvalsh(matrix)
    if extremum == "max":
        return float(np.max(eigvals).real)
    return float(np.min(eigvals).real)


def _pauli_exact_bands(pauli_op) -> list[float]:
    matrix = pauli_op.to_matrix()
    return [float(v.real) for v in np.linalg.eigvalsh(matrix)]


def _export_fermionic_op(model: Model, lattice, n_occ: int, model_params: dict, path: str,
                         observable: str = "E", fermionic_hamiltonian_fn=None):
    fermionic_hamiltonian_fn = fermionic_hamiltonian_fn or model.fermionic_hamiltonian
    op = fermionic_hamiltonian_fn(lattice, **model_params)
    terms = _fermionic_terms(op)

    payload = {
        "format": "qbp_fermionic_op_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model.name,
        "display_name": model.display_name,
        "lattice": list(lattice),
        "spin": model.spin,
        "n_sites": int(op.num_spin_orbitals // model.spin),
        "num_spin_orbitals": int(op.num_spin_orbitals),
        "n_occ": int(n_occ),
        "model_params": model_params,
        "terms": terms,
        "observable": observable,
    }
    if observable != "E":
        obs = model.get_observable(observable)
        if obs.quantum_composite is not None:
            raise ValueError(f"DMRG does not support composite observable '{observable}'.")
        if obs.quantum_operator is None:
            raise ValueError(f"Observable '{observable}' has no fermionic operator for DMRG.")
        observable_op = obs.quantum_operator(model, lattice, **model_params)
        if observable_op is None:
            raise ValueError(f"Observable '{observable}' produced no fermionic operator for DMRG.")
        payload["observable_terms"] = _fermionic_terms(observable_op)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def _default_julia_script():
    return Path(__file__).resolve().parent / "julia-dmrg" / "dmrg_itensor_cli.jl"


def _default_julia_project():
    return Path(__file__).resolve().parent / "julia-dmrg"


def _nersc_julia_path():
    path = Path("/global/common/software/nersc9/julia/1.11.7/bin/julia")
    return str(path) if path.exists() else None


def _sysimage_path():
    """Precompiled sysimage to skip Julia JIT (~80s/cell). QBP_JULIA_SYSIMAGE
    overrides the default location; set it to 'off' to disable."""
    override = os.environ.get("QBP_JULIA_SYSIMAGE")
    if override:
        if override.lower() in ("off", "none", "0"):
            return None
        return override if os.path.exists(override) else None
    default = Path(__file__).resolve().parent / "julia-dmrg" / "dmrg_sysimage.so"
    return str(default) if default.exists() else None


def _run_julia_dmrg(
    hamiltonian_path: str,
    output_path: str,
    *,
    julia: str,
    julia_module: str | None,
    julia_project: str | None,
    nsweeps: int,
    maxdims: str,
    cutoff: float,
    seed: int,
    conserve_qns: bool,
    conserve_sz: bool,
    initial_state: str,
    initial_linkdim: int,
    noise: str,
    mpo_cutoff: float,
    eigsolve_krylovdim: int,
    script_path: str | None,
):
    script = str(Path(script_path) if script_path else _default_julia_script())
    project = julia_project or str(_default_julia_project())
    julia_args = [f"--project={project}"]
    sysimage = _sysimage_path()
    if sysimage:
        julia_args.append(f"--sysimage={sysimage}")
    julia_args += [
        script,
        "--hamiltonian",
        hamiltonian_path,
        "--output",
        output_path,
        "--nsweeps",
        str(nsweeps),
        "--maxdims",
        maxdims,
        "--cutoff",
        str(cutoff),
        "--seed",
        str(seed),
        "--conserve-qns",
        "true" if conserve_qns else "false",
        "--conserve-sz",
        "true" if conserve_sz else "false",
        "--initial-state",
        initial_state,
        "--initial-linkdim",
        str(initial_linkdim),
    ]
    if noise:
        julia_args += ["--noise", noise]
    if mpo_cutoff and mpo_cutoff > 0:
        julia_args += ["--mpo-cutoff", str(mpo_cutoff)]
    if eigsolve_krylovdim and eigsolve_krylovdim > 0:
        julia_args += ["--eigsolve-krylovdim", str(eigsolve_krylovdim)]
    resolved_julia = shutil.which(julia)
    if resolved_julia is None and julia == "julia":
        resolved_julia = _nersc_julia_path()
    if resolved_julia is not None:
        cmd = [resolved_julia, *julia_args]
    elif os.path.isabs(julia):
        cmd = [julia, *julia_args]
    elif julia_module:
        module_cmd = f"module load {shlex.quote(julia_module)} && exec {shlex.quote(julia)} \"$@\""
        cmd = ["bash", "-lc", module_cmd, "qbp-julia", *julia_args]
    else:
        cmd = [julia, *julia_args]
    env = os.environ.copy()
    # Point Julia at a specific depot only if the user asked for one (e.g. a
    # shared HPC depot via QBP_JULIA_DEPOT); otherwise use Julia's default.
    depot = os.environ.get("QBP_JULIA_DEPOT")
    if depot:
        env.setdefault("JULIA_DEPOT_PATH", depot)
    subprocess.run(cmd, check=True, env=env)


@register_method
class DMRGMethod(SimulationMethod):
    METHOD = Method.DMRG
    LABEL = "DMRG"
    PARAM_SPECS = [
        ParamSpec("nsweeps", int, 4, "Number of DMRG sweeps", metavar="N"),
        ParamSpec("maxdims", str, "20,50,100,200", "Comma-separated max bond dims per sweep", metavar="LIST"),
        ParamSpec("cutoff", float, 1e-9, "Truncation cutoff", metavar="F"),
        ParamSpec("seed", int, 1234, "Base RNG seed (offset per cell)", metavar="N"),
        ParamSpec("conserve_qns", bool, True, "Conserve particle number", is_flag=True),
        ParamSpec("conserve_sz", bool, True, "Conserve the spin-z sector", is_flag=True),
        ParamSpec(
            "initial_state", str, "packed",
            "Initial MPS: packed, neel, or random",
            choices=("packed", "neel", "random"),
            metavar="NAME",
        ),
        ParamSpec("initial_linkdim", int, 10,
                  "Bond dimension of the random initial MPS", metavar="N"),
        ParamSpec("noise", str, "",
                  "Comma-separated DMRG noise term per sweep (empty disables)",
                  metavar="LIST"),
        ParamSpec("mpo_cutoff", float, 0.0,
                  "SVD cutoff for MPO construction (0 keeps the ITensor default "
                  "of 1e-15, which perturbs dense 16-qubit Hamiltonians at 1e-8)",
                  metavar="F"),
        ParamSpec("eigsolve_krylovdim", int, 0,
                  "Krylov dimension of the local eigensolver (0 keeps the "
                  "ITensor default of 3)", metavar="N"),
        ParamSpec("julia", str, "julia", "Julia executable", metavar="PATH"),
        ParamSpec("julia_module", str, "julia/1.11.7", "Env module to load if julia not on PATH", metavar="NAME"),
        ParamSpec("julia_project", str, None, "Julia project with ITensors/ITensorMPS/JSON", metavar="PATH"),
        ParamSpec("script_path", str, None, "Override the bundled Julia DMRG script", metavar="PATH"),
    ]
    SUPPORTS_REAL_SPACE = True
    SUPPORTS_BAND_STRUCTURE = True
    SUPPORTS_OPERATOR = True

    def compute_bloch_cell(self, model, k_tuple, cell_params, observable, *,
                           backend, ctx):
        """Compute DMRG for momentum-space (Bloch) Hamiltonian."""
        from qiskit.quantum_info import SparsePauliOp

        H_matrix = model.bloch_hamiltonian(*k_tuple, **cell_params)
        pauli_op = SparsePauliOp.from_operator(H_matrix)
        if pauli_op.num_qubits < 2:
            pauli_op = SparsePauliOp.from_list([
                (f"{pauli}I", complex(coeff))
                for pauli, coeff in zip(pauli_op.paulis, pauli_op.coeffs)
            ])

        cell_tag = f"bloch-{ctx.ix}-{ctx.iy}"
        hamiltonian_path = os.path.join(ctx.tmp_dir, f"{cell_tag}-hamiltonian.json")
        output_path = os.path.join(ctx.raw_dir, f"dmrg-{cell_tag}.json")

        _export_pauli_operator(
            pauli_op, hamiltonian_path, observable=observable,
            label=f"{model.name}_bloch",
        )

        _run_julia_dmrg(
            hamiltonian_path,
            output_path,
            julia=self.julia,
            julia_module=self.julia_module,
            julia_project=self.julia_project,
            nsweeps=self.nsweeps,
            maxdims=self.maxdims,
            cutoff=self.cutoff,
            seed=self.seed + ctx.cell_index,
            conserve_qns=False,
            conserve_sz=False,
            initial_state=_pauli_initial_state(self.initial_state),
            initial_linkdim=self.initial_linkdim,
            noise=self.noise,
            mpo_cutoff=self.mpo_cutoff,
            eigsolve_krylovdim=self.eigsolve_krylovdim,
            script_path=self.script_path,
        )

        with open(output_path) as f:
            raw = json.load(f)

        return {
            "energy": float(raw["energy"]),
            "max_link_dim": raw.get("max_link_dim"),
            "avg_link_dim": raw.get("avg_link_dim"),
            "profile": raw.get("profile"),
        }

    def compute_cell(self, model, lattice, n_occ, cell_params, observable, *,
                     backend, ctx):
        cell_tag = f"cell-x{ctx.ix}-y{ctx.iy}"
        hamiltonian_path = os.path.join(ctx.tmp_dir, f"{cell_tag}-hamiltonian.json")
        output_path = os.path.join(ctx.raw_dir, f"dmrg-{cell_tag}.json")
        spec = _export_fermionic_op(
            model, lattice, n_occ, cell_params, hamiltonian_path, observable,
            fermionic_hamiltonian_fn=ctx.fermionic_hamiltonian_fn,
        )
        _run_julia_dmrg(
            hamiltonian_path,
            output_path,
            julia=self.julia,
            julia_module=self.julia_module,
            julia_project=self.julia_project,
            nsweeps=self.nsweeps,
            maxdims=self.maxdims,
            cutoff=self.cutoff,
            seed=self.seed + ctx.cell_index,
            conserve_qns=self.conserve_qns,
            conserve_sz=self.conserve_sz,
            initial_state=self.initial_state,
            initial_linkdim=self.initial_linkdim,
            noise=self.noise,
            mpo_cutoff=self.mpo_cutoff,
            eigsolve_krylovdim=self.eigsolve_krylovdim,
            script_path=self.script_path,
        )
        with open(output_path) as f:
            raw = json.load(f)
        result = {
            "energy": float(raw["energy"]),
            "hamiltonian_terms": len(spec["terms"]),
            "max_link_dim": raw.get("max_link_dim"),
            "avg_link_dim": raw.get("avg_link_dim"),
            "profile": raw.get("profile"),
            "initial_state": raw.get("initial_state"),
        }
        if observable != "E":
            result["value"] = abs(float(raw["observable_value"]))
            result["observable"] = observable
            result["observable_terms"] = len(spec["observable_terms"])
        return result

    def compute_operator_cell(self, op, *, extremum, backend, label, observable: str = "E"):
        """Compute DMRG cell for a HamLib Pauli operator."""
        if observable != "E":
            raise ValueError(
                "DMRG only supports energy on the --qubit-operator/HamLib path."
            )
        if op.num_qubits < 2:
            return {
                "energy": _pauli_exact_energy(op, extremum=extremum),
                "hamiltonian_terms": len(op.simplify().paulis),
                "profile": {"dmrg": {"skipped": "one_site_pauli_exact"}},
                "initial_state": None,
            }
        sign = -1.0 if extremum == "max" else 1.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            hamiltonian_path = os.path.join(tmp_dir, "hamiltonian.json")
            output_path = os.path.join(tmp_dir, "dmrg_result.json")

            _export_pauli_operator(
                op, hamiltonian_path, observable=observable, scale=sign,
                label=label or "hamlib_operator",
            )

            _run_julia_dmrg(
                hamiltonian_path,
                output_path,
                julia=self.julia,
                julia_module=self.julia_module,
                julia_project=self.julia_project,
                nsweeps=self.nsweeps,
                maxdims=self.maxdims,
                cutoff=self.cutoff,
                seed=self.seed,
                conserve_qns=False,
                conserve_sz=False,
                initial_state=_pauli_initial_state(self.initial_state),
                initial_linkdim=self.initial_linkdim,
                noise=self.noise,
                mpo_cutoff=self.mpo_cutoff,
                eigsolve_krylovdim=self.eigsolve_krylovdim,
                script_path=self.script_path,
            )

            with open(output_path) as f:
                raw = json.load(f)
            with open(hamiltonian_path) as f:
                spec = json.load(f)
            energy = sign * float(raw["energy"])

            return {
                "energy": energy,
                "hamiltonian_terms": len(spec["terms"]),
                "max_link_dim": raw.get("max_link_dim"),
                "avg_link_dim": raw.get("avg_link_dim"),
                "profile": raw.get("profile"),
                "initial_state": raw.get("initial_state"),
            }

    def reduce(self, cell, *, extremum="min", analytic=None):
        if "bands" in cell:
            return cell["bands"]
        return float(cell.get("value", cell["energy"]))

    def parameter_summary(self) -> dict:
        """Exclude Julia/script metadata and fixed parameters from JSON logging."""
        out = super().parameter_summary()
        # Remove internal Julia/script configuration that shouldn't be in public metadata
        for key in ["julia", "julia_module", "julia_project", "script_path",
                    "conserve_qns", "conserve_sz"]:
            out.pop(key, None)
        return out
