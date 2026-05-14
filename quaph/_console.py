from __future__ import annotations

import shlex
import sys
import tempfile
from pathlib import Path

import yaml

from quaph._registry import _MODELS, register_model_from_file, remove_model
from quaph._yaml_model import (
    _QISKIT_OPTIMIZERS,
    YamlModelSpec,
    _make_evaluator,
)


_BANNER = r"""
                                           ,-.----.
    ,----..                                \    /  \    ,---,
   /   /   \                               |   :    \ ,--.' |
  /   .     :            ,--,              |   |  .\ :|  |  :
 .   /   ;.  \         ,'_ /|              .   :  |: |:  :  :
.   ;   /  ` ;    .--. |  | :    ,--.--.   |   |   \ ::  |  |,--.
;   |  ; \ ; |  ,'_ /| :  . |   /       \  |   : .   /|  :  '   |
|   :  | ; | '  |  ' | |  . .  .--.  .-. | ;   | |`-' |  |   /' :
.   |  ' ' ' :  |  | ' |  | |   \__\/: . . |   | ;    '  :  | | |
'   ;  \; /  |  :  | : ;  ; |   ," .--.; | :   ' |    |  |  ' | :
 \   \  ',  . \ '  :  `--'   \ /  /  ,.  | :   : :    |  :  :_:,'
  ;   :      ; |:  ,      .-./;  :   .'   \|   | :    |  | ,'
   \   \ .'`--"  `--`----'    |  ,     .-./`---'.|    `--''
    `---`                      `--`---'      `---`
"""

_HELP = """\
Commands:
  run analytic --model NAME --lattice L [L ...] [...]
  run simulated-ideal --model NAME --lattice L [L ...] [...]
  run simulated-noisy --model NAME --lattice L [L ...] [...]
  plot PATH
  register             Walk through registering a new custom model (writes YAML)
  register --from PATH Register a model from a YAML file
  remove NAME          Permanently remove a registered model
  list                 List registered models
  help                 Show this help
  exit                 Leave the console
"""


def _quaph_version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("quaph")
    except PackageNotFoundError:
        return "0.0.0"


def _print_banner() -> None:
    version = _quaph_version()
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
            _list_models()
        elif head == "register":
            _register_command(argv[1:])
        elif head == "remove":
            _remove_walkthrough(argv[1:])
        else:
            from quaph._cli import main as cli_main
            try:
                cli_main(argv)
            except SystemExit:
                pass
    except KeyboardInterrupt:
        print("\ncancelled.")
    except Exception as e:
        print(f"error: {e}")
    return True


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
    print("This walkthrough writes a YAML file under quaph/models/.")
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

    interaction: list[dict] = []
    if spin == 2 and _prompt_yn("\nAdd a Hubbard-style on-site density-density interaction?"):
        coef = _prompt_expression(f"  coefficient (expression in {sorted(allowed_coef_names)}): ", allowed_coef_names)
        interaction.append({"kind": "density_density_onsite", "coefficient": coef})

    mean_field_correction: str | None = None
    if _prompt_yn("\nProvide a mean-field correction expression?"):
        mf_names = allowed_coef_names | {"n_sites", "n_occ"}
        mean_field_correction = _prompt_expression(
            f"  expression in {sorted(mf_names)}: ", mf_names
        )

    optimizer = None
    if _prompt_yn("\nConfigure a classical optimizer? (otherwise SPSA with @max_iters is used at runtime)", default=True):
        otype = _prompt_choice("  optimizer:", list(_QISKIT_OPTIMIZERS))
        kwargs: dict = {}
        print("  Optimizer kwargs. Enter pairs of <key> <value>. Value may be a number, string,")
        print("  or '@max_iters' to bind the runtime max-iterations arg. Blank key to finish.")
        while True:
            k = _prompt("    key: ")
            if not k:
                break
            v = _prompt_required(f"    value for '{k}': ")
            if v.startswith("@"):
                kwargs[k] = v
            else:
                try:
                    kwargs[k] = int(v)
                except ValueError:
                    try:
                        kwargs[k] = float(v)
                    except ValueError:
                        kwargs[k] = v
        optimizer = {"type": otype, "kwargs": kwargs}

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
    if interaction:
        spec_data["interaction"] = interaction
    if mean_field_correction is not None:
        spec_data["mean_field_correction"] = mean_field_correction
    if optimizer is not None:
        spec_data["optimizer"] = optimizer

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
        "w", suffix=".yaml", delete=False, prefix=f"quaph_{name}_"
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
