# Built-In Models

QBP currently has six built-in tight-binding models. Each is defined declaratively in a YAML file under `qbp/models/` and registered under the short name in the first column below—pass that name as the `model` argument to [`qbp.run`](../api/runners.md) and you are ready to sweep phase diagrams and benchmark quantum methods. Run `qbp list models` to see what is registered in your environment.

The built-in specs are also worked examples: copy one into your own YAML file and edit it to define a variant (see [Defining a Model in YAML](custom-yaml.md)), build the equivalent structure programmatically with [`build_tight_binding_model`](tight-binding-builder.md), or drop to the full Python API for arbitrary Hamiltonians (see [Defining a Model in Python](custom-python.md)). The per-model pages below give each model's Hamiltonian, its parameters, its canonical sweep, and a runnable snippet.

## Overview

| Name | Dim | Spin | Interacting | Phase Diagram | Band Structure |
| --- | --- | --- | --- | --- | --- |
| [`ssh`](ssh.md)                         | 1D | spinless | no  | yes | yes |
| [`haldane`](haldane.md)                 | 2D | spinless | no  | yes | yes |
| [`kane-mele`](kane-mele.md)             | 2D | spinful  | no  | yes | yes |
| [`kane-mele-lc`](kane-mele-lc.md)       | 2D | spinful  | no  | yes | yes |
| [`hubbard`](hubbard.md)                 | 2D | spinful  | yes | yes | —   |
| [`haldane-hubbard`](haldane-hubbard.md) | 2D | spinful  | yes | yes | —   |

```{toctree}
:hidden:
:maxdepth: 1

ssh
haldane
kane-mele
kane-mele-lc
hubbard
haldane-hubbard
```
