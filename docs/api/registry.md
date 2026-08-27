# Registry

QBP keeps a runtime registry mapping each model's `name` to its {py:class}`~qbp.Model`. {py:func}`~qbp.run` and the CLI look models up here by name, and these four functions add, fetch, and remove entries. The built-in models—six models across twelve lattice variants—are registered automatically at import, each under its `<model>-<lattice>` name; the functions below manage your own.

## Fetching a Model

```{eval-rst}
.. autofunction:: qbp.get_model
```

```{code-block} python
import qbp

haldane = qbp.get_model("haldane-honeycomb")
print(haldane.display_name, haldane.n_dims)
```

## Registering a Model

Use {py:func}`~qbp.register_model` for an in-memory {py:class}`~qbp.Model` you built in Python. The registration lives for the current session only.

```{eval-rst}
.. autofunction:: qbp.register_model
```

```{code-block} python
import qbp
from qbp import Model, Method

model = Model(name="ssh_custom", display_name="SSH (custom)", ...)
qbp.register_model(model)

# now runnable by name
qbp.run(model="ssh_custom", method=[Method.ANALYTIC], lattice=(8,),
        x_param="t1", x_range=(0.0, 2.0, 0.05))
```

Use {py:func}`~qbp.register_model_from_file` for a declarative [YAML model](../models/custom-yaml.md). Unlike `register_model`, this **persists**: the file is validated and copied into `qbp/models/`, so the model is available in every future session.

```{eval-rst}
.. autofunction:: qbp.register_model_from_file
```

```{code-block} python
import qbp

model = qbp.register_model_from_file("my_model.yaml")
```

## Removing a Model

```{eval-rst}
.. autofunction:: qbp.remove_model
```

```{code-block} python
import qbp

qbp.remove_model("ssh_custom")
```

`remove_model` unregisters a custom model and deletes its persisted YAML file if one was written; the built-in models cannot be removed.
