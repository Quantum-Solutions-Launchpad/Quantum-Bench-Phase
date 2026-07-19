# Initial Simulation States

Every quantum method in QBP begins from a state it prepares on the qubits, and how that state is built is one of the biggest levers on the method's accuracy. The two quantum methods construct it differently. [VQE](../user-guide/performing-simulation.md) prepares a *parameterized* trial state—the **ansatz**—whose gate parameters the classical optimizer tunes to minimize the measured energy. [IQPE](../user-guide/performing-simulation.md) instead starts from a *fixed* initial state chosen to overlap strongly with the true ground state, so the phase readout has signal to resolve. This page covers both: the ansatz families VQE can use, and the initial-state recipes that seed VQE's ansatz and IQPE's phase estimation.

## Supported Ansätze

Each ansatz `type` is taken from `qiskit.circuit.library`, and QBP forwards your `kwargs` straight to it. The supported values are:

| `type` | Character |
| --- | --- |
| `excitation_preserving` | Conserves particle number; built from two-qubit rotations. The QBP default—well matched to fermionic ground states at fixed filling. |
| `efficient_su2` | Hardware-efficient layers of single-qubit $SU(2)$ rotations and entangling gates. Expressive and shallow, but does *not* conserve particle number. |
| `real_amplitudes` | Real-valued amplitudes only (RY rotations + CX). Cheap and often enough for real Hamiltonians. |
| `pauli_two_design` | A randomized two-design of Pauli rotations, used to study expressibility and barren-plateau behavior. |
| `two_local` | A generic single-rotation-plus-entangler template you fully specify via `kwargs`. |
| `n_local` | The most general template: arbitrary rotation and entanglement blocks. |

## Configuring an Ansatz

An ansatz spec has three parts: the `type`, a `kwargs` dict forwarded to Qiskit object, and an `initial_state_prefix`. In a [YAML model](../models/custom-yaml.md) it reads:

```{code-block} yaml
ansatz:
  type: excitation_preserving
  kwargs:
    mode: fsim
    entanglement: linear
    reps: "@n_layers"
  initial_state_prefix: hartree_fock
```

Any `kwargs` value written as `@<name>` is a **runtime reference**, substituted with the live value when the circuit is built. The ansatz sees `n_qubits`, `n_layers`, `n_occ`, `spin`, and `n_sites`. Writing `reps: "@n_layers"` ties the circuit depth to the run's `method_params` `layers` setting, so the same model can be run shallow or deep without editing the spec.

- **`reps`.** The number of repeated entangling layers. Each rep adds parameters and depth, lifting the expressiveness ceiling while lengthening the optimization. Binding it to `@n_layers` lets you sweep depth from the run call; see [Performing Simulation](../user-guide/performing-simulation.md).
- **`entanglement`.** The connectivity of the entangling block—`linear`, `circular`, `full`, or an explicit pair list. `linear` keeps the circuit shallow and hardware-friendly; `full` is more expressive but far deeper.
- **`mode`** (for `excitation_preserving`). The two-qubit interaction, `fsim` or `iswap`.

## Initial States

The `initial_state_prefix` controls what the ansatz is composed *onto*:

- **`hartree_fock`** prepends `X` gates on the first `n_occ` qubits, preparing the Hartree–Fock reference occupation before the variational block. This gives the optimizer a physically sensible starting point and is the default.
- **`none`** applies the bare ansatz with no prefix—appropriate when the circuit already spans the relevant sector or when you want a neutral starting state.

The [IQPE](../user-guide/performing-simulation.md) simulation method takes a fuller initial-state recipe under the `iqpe_initial_state` key, since its accuracy depends on overlap with the true ground state. Its `type` may be:

| `type` | Meaning |
| --- | --- |
| `hartree_fock` | The Hartree–Fock reference: a proper `HartreeFock` state for spinful models, or the filled-lowest-modes bitstring for spinless ones. |
| `uniform` | An equal superposition (`H` on every qubit). |
| `computational_zero` | The all-zero state. |
| `vqe_informed` | Run a short VQE first and use its optimized circuit as the initial state; configured with a nested `vqe_ansatz`, `vqe_optimizer`, `vqe_n_layers`, and `vqe_max_iters`. |

## Defaults and Where to Set It

If a model specifies no ansatz, QBP falls back to an `excitation_preserving` circuit with `fsim` gates and `linear` entanglement, preceded by the Hartree–Fock `X`-gate prefix on the first `n_occ` qubits. This is a sensible particle-number-conserving default for the built-in fermionic models.

The ansatz is a property of the model, so set it where the model is defined:

- **YAML models**—add the `ansatz` block shown above; see the [YAML schema](../models/custom-yaml.md).
- **Python models**—pass a `get_vqe_ansatz` function `(n_qubits, n_layers, n_occ, spin) -> QuantumCircuit` to the [`Model`](../api/model.md) constructor.
- **HamLib Hamiltonians**, which have no model object, take the ansatz per run instead—via `method_params[Method.VQE] = {"ansatz": {...}}` or the `--vqe-ansatz` / `--vqe-ansatz-kwarg` CLI flags.

Whichever route you use, the supported `type` values are exactly those listed above.
