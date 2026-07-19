# Exceptions

QBP raises {py:exc}`~qbp.ModelCapabilityError` when a {py:class}`~qbp.Model` is asked for something it does not implement—most often a [band-structure](../more-examples/band-structure.md) run on a model without a `bloch_hamiltonian`, an observable the model lacks, or an [investigation](investigations.md) whose model requirements aren't met. Ordinary bad arguments raise the usual `ValueError`/`TypeError`; this exception specifically signals a missing *capability*.

```{eval-rst}
.. autoexception:: qbp.ModelCapabilityError
   :members:
```

## Example

Catch it to fall back gracefully when a capability may be absent—for example, probing whether a model supports band structure:

```{code-block} python
import qbp
from qbp import Method

try:
    result = qbp.run(
        model="kane-mele",          # uses spin_channels, so no Bloch Hamiltonian
        method=[Method.ANALYTIC],
        x_param="k",
        x_range=(-3.14, 3.14, 0.05),
    )
except qbp.ModelCapabilityError as err:
    print(f"Band structure unavailable: {err}")
```
