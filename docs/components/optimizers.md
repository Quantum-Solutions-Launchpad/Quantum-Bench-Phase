# Optimizers

[VQE](../user-guide/performing-simulation.md) is a hybrid loop: the quantum device measures the energy of a trial state, and a classical **optimizer** proposes the next set of ansatz parameters to try. The optimizer decides how the parameter search moves—whether it estimates gradients, how it copes with the sampling noise in each energy estimate, and how quickly it converges. QBP exposes the optimizers from `qiskit_algorithms.optimizers`, so you pick one per model and pass its constructor arguments through.

## Supported Optimizers

Each `type` names a class in `qiskit_algorithms.optimizers`; your `kwargs` are forwarded to it verbatim.

| `type` | Kind | Notes |
| --- | --- | --- |
| `SPSA` | Gradient-free (stochastic) | The default. Two energy evaluations per step regardless of parameter count; tolerant of sampling noise. |
| `QNSPSA` | Gradient-free (stochastic) | Quantum-natural SPSA; adds curvature information for faster convergence, still noise-tolerant. |
| `COBYLA` | Gradient-free | Linear-approximation trust region; robust and simple, a good noisy fallback. |
| `NELDER_MEAD` | Gradient-free | Simplex search; no gradients, but slows in high dimensions. |
| `POWELL` | Gradient-free | Direction-set method along conjugate directions. |
| `NFT` | Gradient-free | Nakanishi–Fujii–Todo sequential sinusoidal fitting, tailored to parameterized circuits. |
| `L_BFGS_B` | Gradient-based | Quasi-Newton; fast and accurate on smooth, low-noise landscapes. |
| `SLSQP` | Gradient-based | Sequential least-squares; converges quickly when gradients are reliable. |
| `CG` | Gradient-based | Nonlinear conjugate gradient. |
| `ADAM` | Gradient-based | Adaptive-moment stochastic gradient; familiar from machine learning. |
| `GradientDescent` | Gradient-based | Plain gradient descent with a fixed or scheduled step. |
| `AQGD` | Gradient-based | Analytic quantum gradient descent using parameter-shift gradients. |
| `P_BFGS` | Gradient-based | Parallelized BFGS that spreads gradient evaluations across processes. |

## Configuring an Optimizer

An optimizer spec is a `type` plus a `kwargs` dict forwarded to the Qiskit class constructor. In a [YAML model](../models/custom-yaml.md):

```{code-block} yaml
optimizer:
  type: SPSA
  kwargs:
    maxiter: "@max_iters"
```

A `kwargs` value written as `@<name>` is a **runtime reference**, substituted when the optimizer is built. The optimizer sees a single runtime name, `max_iters`, which carries the run's `method_params` VQE `iters` setting. Binding `maxiter: "@max_iters"` lets you control the iteration budget from the run call rather than editing the model. Any other constructor argument the class accepts—`learning_rate`, `perturbation`, `tol`, and so on—can be given as a literal alongside it.

## Choosing an Optimizer

The right choice depends mainly on whether energy estimates are noisy:

- **Noisy runs** (a noise model or real hardware, where every energy is a shot-estimated random variable) favor the gradient-free, noise-tolerant methods. `SPSA` and `QNSPSA` are the standard picks because their two-evaluation stochastic step degrades gracefully under sampling noise; `COBYLA` is a reasonable gradient-free fallback. Gradient-based methods tend to chase the noise and stall.
- **Ideal runs** (a noise-free statevector simulator, where the energy is exact) may more strongly favor gradient-based methods. `L_BFGS_B`, `SLSQP`, and `CG` converge faster and deeper than SPSA when the landscape is smooth and the gradients are trustworthy. `COBYLA` also does well here.

See [Performing Simulation](../user-guide/performing-simulation.md) for how the optimizer interacts with the ansatz depth and iteration budget, and [Incorporating Quantum Hardware](../user-guide/incorporating-quantum-hardware.md) for the noisy-backend side.

## Defaults and Where to Set It

If a model specifies no optimizer, QBP uses `SPSA(maxiter=max_iters)`—a default particularly suited for many-body Hamiltonians that works on ideal and noisy runs alike. The optimizer is a property of the model:

- **YAML models**—add the `optimizer` block shown above; see the [YAML schema](../models/custom-yaml.md).
- **Python models**—pass a `get_optimizer` function `(max_iters) -> Optimizer` to the [`Model`](../api/model.md) constructor.
- **HamLib Hamiltonians**, which have no model object, take the optimizer per run—via `method_params[Method.VQE] = {"optimizer": {...}}` or the `--vqe-optimizer` / `--vqe-optimizer-kwarg` CLI flags.
