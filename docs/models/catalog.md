# Built-In Models

```{eval-rst}
.. todo:: One-paragraph framing of QBP's built-in catalog: where the YAML
   files live (``qbp/models/``), how to override or extend them, and a
   short note pointing readers to the per-model pages below for parameters
   and runnable snippets.
```

## Overview

| Name | Dim | Spin | Interacting | Phase diagram | Band structure |
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
