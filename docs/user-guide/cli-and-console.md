# CLI and Console

Everything you can do with the Python API you can also do from the command line. QBP installs a `qbp` executable that mirrors [`qbp.run`](../api/runners.md) flag-for-flag, plus a few housekeeping commands for managing models and re-plotting saved runs. Running `qbp` with no arguments drops you into an interactive console that speaks the same command language. Use whichever fits the moment: the Python API for scripting and analysis, the CLI for one-off runs and shell pipelines, the console for exploring interactively.

## The `qbp run` Command

`qbp run` is the CLI face of `qbp.run`. The Python call from [Performing Simulation](performing-simulation.md) translates directly:

```{code-block} console
$ qbp run --model haldane-honeycomb --method analytic vqe dmrg \
    --lattice 2 2 \
    --x-param n_occ \
    --y-param t2 --y-range 0.0 1.0 0.1 \
    --t1 1.0 --phi 0.7853981633974483 --M 0.0 \
    --vqe-iters 10000 --vqe-layers 5 --vqe-reps 10 \
    --dmrg-nsweeps 4 --dmrg-maxdims 20,50,100,200 --dmrg-cutoff 1e-9
```

The mapping is mechanical:

- **`--model`** and **`--method`** name the model and one or more methods, exactly as in Python.
- **`--lattice`** takes one integer per dimension; omit it for a band-structure run.
- **`--x-param` / `--x-range`** and **`--y-param` / `--y-range`** set the sweep axes. A range is `MIN MAX STEP` (the step may be dropped only on the `--qubit-operator` path).
- **Model parameters get their own flags.** Because each model declares its own parameters, `qbp run --model haldane-honeycomb` grows `--t1`, `--t2`, `--phi`, and `--M` flags on the fly. Every parameter that isn't a sweep axis must be given a value.
- **Method parameters are prefixed by method.** `--vqe-iters`, `--vqe-layers`, `--vqe-reps`, `--iqpe-time`, `--iqpe-trot`, `--iqpe-iters`, and so on—one flag per entry you'd put in `method_params`.
- **`--observable`**, **`--heatmap`**, **`--backend`**, **`--log-path`**, and **`--plot-path`** carry the same meaning as their keyword counterparts. Add `--backend FakeSherbrooke` for local noise, or a device name for hardware (see [Incorporating Quantum Hardware](incorporating-quantum-hardware.md)).

To sweep [HamLib](https://quantum-journal.org/papers/q-2024-12-11-1559/) problem Hamiltonians instead of a built-in model, swap `--model` for `--qubit-operator SOURCE`, where the source is a local `.h5`/`.hdf5` file, a `.zip` containing one, or an `http(s)` URL. The two are mutually exclusive. Choose sweep axes by naming key tokens (`--x-param h --y-param Lx`), and narrow a multi-family source to one Hamiltonian family with `--select 1D,grid,pbc`.

## The `qbp plot` Command

`qbp plot` reloads a saved run and redraws it—the CLI equivalent of `qbp.load_result(path).plot()`:

```{code-block} console
$ qbp plot runs/haldane/2x2/simulated-ideal-3d-n_occ-vs-t2.json
$ qbp plot runs/haldane/2x2/simulated-ideal-3d-n_occ-vs-t2.json --output figure.pdf
```

It accepts `--output` for a target file, `--hide-plot` and `--hide-legend` to shape the figure, and `--diff` (with `--diff-format`) to add the method-vs-method difference plots described in [Results and Plotting](results-and-plotting.md). This is the fastest way to re-render a run you computed earlier without touching Python.

## Managing Models

Three commands cover the model registry:

```{code-block} console
$ qbp list models                        # every registered model
$ qbp list observables --model haldane-honeycomb    # observables a model exposes
$ qbp register --from my-model.yaml       # register a model from a YAML spec
$ qbp remove my-model                     # permanently remove a registered model
```

`qbp register --from PATH` reads a YAML model spec and registers it in one shot. Running `qbp register` with no path instead launches an interactive walkthrough that asks for the model's parameters, sublattices, and Hamiltonian terms one at a time and writes the YAML for you—handy when you don't want to author the spec by hand. See [Custom Models (YAML)](../models/custom-yaml.md) for the spec format.

## The Interactive Console

Running `qbp` with no arguments opens a REPL:

```{code-block} console
$ qbp
```

The console accepts the same commands you'd type after `qbp`—`run ...`, `plot ...`, `list models`, `register`, `remove`, plus `help` and `exit`—but keeps the library loaded between commands, so there's no per-invocation import cost. That makes it the most comfortable place to iterate: tweak a parameter, rerun, adjust an axis, rerun again, all without leaving the session or re-importing QBP each time. Type `help` at the prompt for the full command list.

## Choosing an Interface

All three interfaces call the same `qbp.run` underneath, so the choice is about ergonomics rather than capability:

- **Python API** — best for scripting, batch sweeps, and anything where you want to hold onto the `RunResult` and analyze the grids yourself.
- **`qbp` CLI** — best for one-off runs, shell scripts, cluster job submissions, and pipelines where a single command is more convenient than a Python file.
- **Interactive console** — best for exploration, when you're iterating quickly on parameters and want the library to stay warm between runs.
