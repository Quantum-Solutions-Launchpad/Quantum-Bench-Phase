from __future__ import annotations

import shlex
import sys
import tempfile
from pathlib import Path

import yaml

from qbp._registry import _MODELS, register_model_from_file, remove_model
from qbp._yaml_model import (
    _QISKIT_ANSATZES,
    _QISKIT_MAPPERS,
    _QISKIT_OPTIMIZERS,
    _INITIAL_STATES,
    YamlModelSpec,
    _make_evaluator,
)


_BANNER = r"""                        
                  ..                  
            . uW8"                    
            `t888        .d``         
    .u@u     8888   .    @8Ne.   .u   
 .zWF8888bx  9888.z88N   %8888:u@88N  
.888  9888   9888  888E   `888I  888. 
I888  9888   9888  888E    888I  888I 
I888  9888   9888  888E    888I  888I 
I888  9888   9888  888E  uW888L  888' 
`888Nx?888  .8888  888" '*88888Nu88P  
 "88" '888   `%888*%"   ~ '88888F`    
       88E      "`         888 ^      
       98>                 *8E        
       '8                  '8>        
        `                   "                                 
"""

_HELP = """\
Commands:
  run --model NAME --method M [M ...] --lattice L [L ...] [...]
                       Run one or more simulation methods over a parameter sweep.
                       Methods: analytic, vqe, iqpe, dmrg (choose any combination).
                       Per-method flags are prefixed, e.g. --vqe-iters 200
                       --vqe-layers 2 --iqpe-time 0.2 --dmrg-nsweeps 4. Add
                       --backend NAME to run vqe/iqpe under a noise model
                       (e.g. FakeSherbrooke), on a real IBM device (ibm_brisbane,
                       least_busy), or on an IQM Resonance device (iqm_emerald,
                       iqm_garnet, iqm_sirius).
    OR
  run --qubit-operator SOURCE --method M [M ...] [--extremum min|max] [...]
                       Sweep a HamLib HDF5 file's Hamiltonians instead of a
                       registered model (analytic/vqe/iqpe only). SOURCE is a local
                       .h5/.hdf5 file, a local .zip archive containing one, or an
                       http(s) URL to either. Choose sweep axes with --x-param/
                       --y-param naming key tokens (e.g. --x-param h --y-param Lx);
                       a range may omit STEP to use every available value. Narrow
                       multi-family sources with --select 1D,grid,pbc. With no axes
                       it sweeps all keys by instance index.
  plot PATH
  register             Walk through registering a new custom model (writes YAML)
  register --from PATH Register a model from a YAML file
  remove NAME          Permanently remove a registered model
  list models          List registered models
  list observables --model NAME
                       List observables exposed by a model
  help                 Show this help
  exit                 Leave the console
"""


def _qbp_version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("qbp")
    except PackageNotFoundError:
        return "0.0.0"


def _print_banner() -> None:
    version = _qbp_version()
    tagline = f'v{version} — type "help" if you don\'t know what to do.'
    if not sys.stdout.isatty():
        print(_BANNER)
        print(tagline)
        return
    try:
        from rich.console import Console
        from rich.text import Text
    except ImportError:
        print(_BANNER)
        print(tagline)
        return

    lines = _BANNER.splitlines()
    start = (0x00, 0xD9, 0xFF)
    end = (0xFF, 0x5F, 0xD2)
    n = max(len(lines) - 1, 1)
    text = Text()
    for i, line in enumerate(lines):
        t = i / n
        r = round(start[0] + (end[0] - start[0]) * t)
        g = round(start[1] + (end[1] - start[1]) * t)
        b = round(start[2] + (end[2] - start[2]) * t)
        text.append(line + "\n", style=f"bold #{r:02x}{g:02x}{b:02x}")
    console = Console(color_system="truecolor", highlight=False)
    console.print(text)
    print(tagline)


_PROMPT_TTY = (
    "\x1b[38;2;0;217;255m>"
    "\x1b[38;2;127;156;232m>"
    "\x1b[38;2;255;95;210m>"
    "\x1b[0m "
)


def run_console(initial_command: str | None = None) -> int:
    interactive = sys.stdin.isatty()
    if interactive:
        _print_banner()
    prompt = _PROMPT_TTY if interactive else ""
    if initial_command:
        if not _handle_line(initial_command):
            return 0
    while True:
        try:
            line = input(prompt).strip()
        except EOFError:
            if interactive:
                print()
            return 0
        except KeyboardInterrupt:
            if interactive:
                print()
            return 0
        if not line or line.startswith("#"):
            continue
        if not _handle_line(line):
            return 0


def _handle_line(line: str) -> bool:
    if line in ("exit", "quit"):
        return False
    try:
        argv = shlex.split(line)
    except ValueError as e:
        print(f"error: {e}")
        return True
    if not argv:
        return True
    head = argv[0]
    try:
        if head == "help":
            print(_HELP, end="")
        elif head == "list":
            _list_command(argv[1:])
        elif head == "register":
            _register_command(argv[1:])
        elif head == "remove":
            _remove_walkthrough(argv[1:])
        else:
            from qbp._cli import main as cli_main
            try:
                cli_main(argv)
            except SystemExit:
                pass
    except KeyboardInterrupt:
        print("\ncancelled.")
    except Exception as e:
        print(f"error: {e}")
    return True


def _list_command(args: list[str]) -> None:
    if not args:
        print("usage: list models | list observables --model NAME")
        return
    target = args[0]
    if target == "models":
        _list_models()
        return
    if target == "observables":
        model_name = None
        rest = args[1:]
        if len(rest) == 2 and rest[0] == "--model":
            model_name = rest[1]
        elif len(rest) == 1:
            model_name = rest[0]
        if not model_name:
            print("usage: list observables --model NAME")
            return
        from qbp._registry import get_model
        try:
            model = get_model(model_name)
        except ValueError as e:
            print(f"error: {e}")
            return
        width = max(len(n) for n in model.observables)
        for name in sorted(model.observables):
            obs = model.observables[name]
            print(f"  {name.ljust(width)}  {obs.display_name}")
        return
    print(f"unknown list target '{target}'; choose 'models' or 'observables'.")


def _list_models() -> None:
    if not _MODELS:
        print("(no models registered)")
        return
    width = max(len(n) for n in _MODELS)
    for name in sorted(_MODELS):
        m = _MODELS[name]
        print(f"  {name.ljust(width)}  {m.display_name}")


def _remove_walkthrough(args: list[str]) -> None:
    if len(args) != 1:
        print("usage: remove <model-name>")
        return
    remove_model(args[0])


def _register_command(args: list[str]) -> None:
    if not args:
        _register_walkthrough()
        return
    if len(args) == 2 and args[0] == "--from":
        model = register_model_from_file(args[1])
        print(f"Registered '{model.name}'.")
        return
    print("usage: register | register --from PATH")


def _prompt(prompt: str) -> str:
    return input(prompt).strip()


def _prompt_required(prompt: str) -> str:
    while True:
        val = _prompt(prompt)
        if val and val.lower() != "skip":
            return val
        print("  (this field is required)")


def _prompt_int(prompt: str, allowed: tuple[int, ...] | None = None, minimum: int | None = None) -> int:
    while True:
        try:
            v = int(_prompt_required(prompt))
        except ValueError:
            print("  (must be an integer)")
            continue
        if allowed is not None and v not in allowed:
            print(f"  (must be one of {allowed})")
            continue
        if minimum is not None and v < minimum:
            print(f"  (must be >= {minimum})")
            continue
        return v


def _prompt_yn(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    raw = _prompt(prompt + suffix).lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def _validate_expression(expr: str, allowed_names: set[str]) -> str | None:
    aeval = _make_evaluator({n: 1.0 for n in allowed_names})
    aeval(expr, show_errors=False)
    if aeval.error:
        first = aeval.error[0].get_error()
        return f"{first[0]}: {first[1]}"
    return None


def _prompt_expression(prompt: str, allowed_names: set[str]) -> str:
    while True:
        expr = _prompt_required(prompt)
        err = _validate_expression(expr, allowed_names)
        if err is None:
            return expr
        print(f"  invalid: {err}")


def _coerce_kwarg_value(v: str):
    if v.startswith("@"):
        return v
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _prompt_kwargs(label: str, runtime_args: tuple[str, ...]) -> dict:
    kwargs: dict = {}
    bindings = ", ".join(f"@{a}" for a in runtime_args) if runtime_args else "(none)"
    print(f"  {label.capitalize()} kwargs. Enter pairs of <key> <value>. Value may be a number,")
    print(f"  string, or one of the runtime bindings {bindings}. Blank key to finish.")
    while True:
        k = _prompt("    key: ")
        if not k:
            return kwargs
        v = _prompt_required(f"    value for '{k}': ")
        kwargs[k] = _coerce_kwarg_value(v)


def _prompt_factory_block(
    *, prompt: str, label: str, choices: list[str],
    runtime_args: tuple[str, ...], default_yes: bool,
) -> dict | None:
    if not _prompt_yn(prompt, default=default_yes):
        return None
    otype = _prompt_choice(f"  {label}:", choices)
    kwargs = _prompt_kwargs(label, runtime_args)
    return {"type": otype, "kwargs": kwargs}


def _prompt_observables_block(param_names: set[str], n_dims: int) -> dict | None:
    if not _prompt_yn(
        "\nDeclare extra observables besides energy? (e.g. gap, double_occupancy)",
        default=False,
    ):
        return None
    momentum = {1: ("k",), 2: ("kx", "ky"), 3: ("kx", "ky", "kz")}[n_dims]
    analytic_names = set(param_names) | {
        "eigvals", "eigvecs", "H", "rho", "n_occ", "n_sites", "lattice",
    }
    bloch_names = set(param_names) | {"eigvals", "eigvecs", "H", "n_bands"} | set(momentum)
    out: dict[str, dict] = {}
    while True:
        oname = _prompt("  observable name (blank to finish): ")
        if not oname:
            return out or None
        if oname in out or oname == "E":
            print(f"  '{oname}' already declared; pick another.")
            continue
        display = _prompt_required(f"  display label for '{oname}' (e.g. '\\Delta'): ")
        analytic_expr = _prompt_expression(
            f"  analytic expression in {sorted(analytic_names)}: ", analytic_names
        )
        entry: dict = {"display_name": display, "analytic": analytic_expr}
        if _prompt_yn("  also provide an analytic_bloch expression?", default=False):
            entry["analytic_bloch"] = _prompt_expression(
                f"    expression in {sorted(bloch_names)}: ", bloch_names
            )
        out[oname] = entry


def _prompt_ansatz_block() -> dict | None:
    if not _prompt_yn(
        "\nConfigure a VQE ansatz? (otherwise excitation_preserving "
        "with fsim/linear + HF X-prefix is used)",
        default=False,
    ):
        return None
    otype = _prompt_choice("  ansatz:", list(_QISKIT_ANSATZES))
    kwargs = _prompt_kwargs(
        "ansatz",
        runtime_args=("n_qubits", "n_layers", "n_occ", "spin", "n_sites"),
    )
    prefix = _prompt_choice(
        "  initial_state_prefix (X gates on first n_occ qubits before the ansatz body):",
        ["hartree_fock", "none"],
    )
    return {"type": otype, "kwargs": kwargs, "initial_state_prefix": prefix}


def _prompt_iqpe_initial_state_block() -> dict | None:
    if not _prompt_yn(
        "\nConfigure an IQPE initial state? (otherwise hartree_fock for fermionic, "
        "uniform superposition for operator path)",
        default=False,
    ):
        return None
    kind = _prompt_choice("  initial state type:", list(_INITIAL_STATES))
    d: dict = {"type": kind}
    if kind == "vqe_informed":
        print("  Configure the VQE run used to prepare the initial state.")
        ansatz_type = _prompt_choice("    ansatz:", list(_QISKIT_ANSATZES))
        kwargs = _prompt_kwargs(
            "initial-vqe ansatz",
            runtime_args=("n_qubits", "n_layers", "n_occ", "spin", "n_sites"),
        )
        prefix = _prompt_choice(
            "    initial_state_prefix (X gates on first n_occ qubits before ansatz body):",
            ["hartree_fock", "none"],
        )
        d["vqe_ansatz"] = {"type": ansatz_type, "kwargs": kwargs, "initial_state_prefix": prefix}
        raw_layers = _prompt("    vqe_n_layers [1]: ").strip()
        if raw_layers:
            d["vqe_n_layers"] = int(raw_layers)
        raw_iters = _prompt("    vqe_max_iters [100]: ").strip()
        if raw_iters:
            d["vqe_max_iters"] = int(raw_iters)
    return d


def _prompt_choice(prompt: str, options: list[str]) -> str:
    while True:
        print(prompt)
        for i, opt in enumerate(options):
            print(f"  {i + 1}. {opt}")
        raw = _prompt_required("  > ")
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        if raw in options:
            return raw
        print("  (invalid choice; pick a number or name)")


def _prompt_offsets(n_dims: int) -> list[list[int]]:
    print(f"  Enter offset vectors (each a comma-separated tuple of {n_dims} ints).")
    print(f"  Blank line to finish.")
    offsets: list[list[int]] = []
    while True:
        raw = _prompt("    offset: ")
        if not raw:
            if not offsets:
                print("    (need at least one offset)")
                continue
            return offsets
        try:
            parts = [int(p) for p in raw.split(",")]
        except ValueError:
            print("    (must be comma-separated ints)")
            continue
        if len(parts) != n_dims:
            print(f"    (need exactly {n_dims} components)")
            continue
        offsets.append(parts)


def _prompt_spin_channels(spin: int) -> list[str] | None:
    if spin != 2:
        return None
    raw = _prompt("  spin_channels [up,down / up / down] (blank = both): ").lower()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    bad = [p for p in parts if p not in ("up", "down")]
    if bad:
        print(f"  (ignoring unknown spin channels: {bad}); using both")
        return None
    return parts


def _register_walkthrough() -> None:
    print("\n--- Register a custom model ---")
    print("This walkthrough writes a YAML file under qbp/models/.")
    print("Type 'skip' to omit any optional field.\n")

    while True:
        name = _prompt_required("Model name (unique identifier, e.g. 'ssh'): ")
        if name in _MODELS:
            print(f"  a model named '{name}' is already registered; pick another.")
            continue
        break
    display_name = _prompt_required("Display name (human-readable, e.g. 'SSH'): ")
    spin = _prompt_int("spin (1 = spinless, 2 = with spin): ", allowed=(1, 2))
    n_dims = _prompt_int("n_dims (lattice spatial dimensionality 1/2/3): ", allowed=(1, 2, 3))

    while True:
        raw = _prompt_required(f"lattice_shape (comma-separated axis names, {n_dims} entries, e.g. 'Lx,Ly'): ")
        lattice_shape = [s.strip() for s in raw.split(",") if s.strip()]
        if len(lattice_shape) == n_dims:
            break
        print(f"  (need exactly {n_dims} entries)")

    sites_per_cell = _prompt_int("sites_per_cell (atoms per unit cell): ", minimum=1)

    while True:
        raw = _prompt_required(f"sublattice names (comma-separated, {sites_per_cell} entries, e.g. 'A,B'): ")
        sublattices = [s.strip() for s in raw.split(",") if s.strip()]
        if len(sublattices) == sites_per_cell:
            break
        print(f"  (need exactly {sites_per_cell} entries)")

    print("\nParameter declarations (e.g. t1, t2, phi, M).")
    print("  Enter pairs of <name> <display_label>. Blank name to finish.")
    parameters: dict[str, dict] = {}
    while True:
        name_in = _prompt("  parameter name: ")
        if not name_in:
            if not parameters:
                print("  (at least one parameter is required)")
                continue
            break
        if name_in in parameters:
            print(f"  '{name_in}' already declared; pick another or blank to finish.")
            continue
        label = _prompt_required(f"  display label for '{name_in}' (e.g. 't_1', '\\phi'): ")
        parameters[name_in] = {"label": label}

    allowed_coef_names = set(parameters)

    terms: list[dict] = []
    print("\nHamiltonian terms.")
    while True:
        kind = _prompt(
            "  Add term [onsite/hopping/done]: "
        ).lower()
        if kind in ("done", "", "skip"):
            break
        if kind == "onsite":
            sublattice = _prompt_choice("  sublattice:", sublattices)
            coef = _prompt_expression(f"  coefficient (expression in {sorted(allowed_coef_names)}): ", allowed_coef_names)
            term = {"kind": "onsite", "sublattice": sublattice, "coefficient": coef}
            sc = _prompt_spin_channels(spin)
            if sc is not None:
                term["spin_channels"] = sc
            terms.append(term)
        elif kind == "hopping":
            src = _prompt_choice("  from sublattice:", sublattices)
            dst = _prompt_choice("  to sublattice:", sublattices)
            offsets = _prompt_offsets(n_dims)
            coef = _prompt_expression(f"  coefficient (expression in {sorted(allowed_coef_names)}): ", allowed_coef_names)
            herm = _prompt_yn("  add hermitian partner?", default=True)
            term = {
                "kind": "hopping",
                "from": src,
                "to": dst,
                "offsets": offsets,
                "coefficient": coef,
                "hermitian_partner": herm,
            }
            sc = _prompt_spin_channels(spin)
            if sc is not None:
                term["spin_channels"] = sc
            terms.append(term)
        else:
            print(f"  unknown term kind '{kind}'; choose 'onsite' or 'hopping'.")

    if spin == 2 and _prompt_yn("\nAdd a Hubbard-style on-site density-density interaction?"):
        coef = _prompt_expression(f"  coefficient (expression in {sorted(allowed_coef_names)}): ", allowed_coef_names)
        terms.append({"kind": "density_density", "on": "*", "coefficient": coef})

    optimizer = _prompt_factory_block(
        prompt="\nConfigure a classical optimizer? (otherwise SPSA with @max_iters is used at runtime)",
        label="optimizer",
        choices=list(_QISKIT_OPTIMIZERS),
        runtime_args=("max_iters",),
        default_yes=True,
    )

    mapper = _prompt_factory_block(
        prompt="\nConfigure a qubit mapper? (otherwise JordanWignerMapper is used)",
        label="mapper",
        choices=list(_QISKIT_MAPPERS),
        runtime_args=("n_sites", "spin", "n_occ", "num_particles"),
        default_yes=False,
    )

    ansatz = _prompt_ansatz_block()

    iqpe_initial_state = _prompt_iqpe_initial_state_block()

    observables = _prompt_observables_block(allowed_coef_names, n_dims)

    spec_data: dict = {
        "name": name,
        "display_name": display_name,
        "spin": spin,
        "n_dims": n_dims,
        "lattice_shape": lattice_shape,
        "sites_per_cell": sites_per_cell,
        "sublattices": sublattices,
        "parameters": parameters,
        "terms": terms,
    }
    if optimizer is not None:
        spec_data["optimizer"] = optimizer
    if mapper is not None:
        spec_data["mapper"] = mapper
    if ansatz is not None:
        spec_data["ansatz"] = ansatz
    if iqpe_initial_state is not None:
        spec_data["iqpe_initial_state"] = iqpe_initial_state
    if observables is not None:
        spec_data["observables"] = observables

    try:
        YamlModelSpec.model_validate(spec_data)
    except Exception as e:
        print(f"\nvalidation failed: {e}")
        print("aborted; nothing written.")
        return

    yaml_text = yaml.safe_dump(spec_data, sort_keys=False, allow_unicode=True)
    print("\n--- Generated YAML ---")
    print(yaml_text)
    if not _prompt_yn("Write this model?", default=True):
        print("aborted; nothing written.")
        return

    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, prefix=f"qbp_{name}_"
    ) as tmp:
        tmp.write(yaml_text)
        tmp_path = Path(tmp.name)
    try:
        register_model_from_file(tmp_path)
    except Exception as e:
        print(f"error: {e}")
        return
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"Registered '{name}'.")
