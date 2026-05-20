# Built-In Models

```{eval-rst}
.. todo:: One-paragraph framing of QuaPh's built-in catalog: where the YAML
   files live (``quaph/models/``), how to override or extend them, and a
   short note pointing readers to the per-model pages below for parameters
   and runnable snippets.
```

## Overview

| Name | Dim | Spin | Interacting | Phase diagram | Band structure |
| --- | --- | --- | --- | --- | --- |
| [`ssh`](ssh.md)                               | 1D | spinless | no  | yes | yes |
| [`rice-mele`](rice-mele.md)                   | 1D | spinless | no  | yes | yes |
| [`qwz`](qwz.md)                               | 2D | spinless | no  | yes | yes |
| [`haldane`](haldane.md)                       | 2D | spinless | no  | yes | yes |
| [`bhz`](bhz.md)                               | 2D | spinful  | no  | yes | yes |
| [`kane-mele`](kane-mele.md)                   | 2D | spinful  | no  | yes | yes |
| [`lieb`](lieb.md)                             | 2D | spinless | no  | yes | yes |
| [`t-V`](t-V.md)                               | 1D | spinless | yes | yes | —   |
| [`hubbard`](hubbard.md)                       | 2D | spinful  | yes | yes | —   |
| [`extended-hubbard`](extended-hubbard.md)     | 2D | spinful  | yes | yes | —   |
| [`ionic-hubbard`](ionic-hubbard.md)           | 2D | spinful  | yes | yes | —   |
| [`hubbard-triangular`](hubbard-triangular.md) | 2D | spinful  | yes | yes | —   |
| [`haldane-hubbard`](haldane-hubbard.md)       | 2D | spinful  | yes | yes | —   |
| [`kane-mele-hubbard`](kane-mele-hubbard.md)   | 2D | spinful  | yes | yes | —   |

```{toctree}
:hidden:
:maxdepth: 1

ssh
rice-mele
qwz
haldane
bhz
kane-mele
lieb
t-V
hubbard
extended-hubbard
ionic-hubbard
hubbard-triangular
haldane-hubbard
kane-mele-hubbard
```
