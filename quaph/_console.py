from __future__ import annotations

import shlex
import sys

from quaph._model import Model
from quaph._registry import _MODELS, register_model, remove_model


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
  run analytic --model NAME --n-sites N [...]
  run simulated-ideal --model NAME --n-sites N [...]
  run simulated-noisy --model NAME --n-sites N [...]
  plot PATH
  register             Walk through registering a new custom model
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
            _register_walkthrough()
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


def _prompt(prompt: str) -> str:
    return input(prompt).strip()


def _prompt_required(prompt: str) -> str:
    while True:
        val = _prompt(prompt)
        if val and val.lower() != "skip":
            return val
        print("  (this field is required)")


def _parse_number(s: str):
    try:
        if "." in s or "e" in s or "E" in s:
            return float(s)
        return int(s)
    except ValueError:
        return float(s)


def _read_paste_block(label: str, signature_hint: str) -> str | None:
    print(f"\n{label} (optional)")
    print(f"  Expected signature: {signature_hint}")
    print("  Type 'skip' to omit this callable, or paste Python source and finish")
    print("  with a single line containing only END.")
    first = input("  > ").rstrip()
    if first.strip().lower() == "skip":
        return None
    lines: list[str] = [first]
    while True:
        try:
            ln = input()
        except EOFError:
            break
        if ln.strip() == "END":
            break
        lines.append(ln)
    src = "\n".join(lines).strip()
    if not src:
        return None
    return src


def _exec_callable(src: str, field_name: str):
    ns: dict = {}
    exec(compile(src, f"<paste:{field_name}>", "exec"), ns)
    funcs = [
        v for k, v in ns.items()
        if callable(v) and not k.startswith("_") and getattr(v, "__module__", None) is None
    ]
    if not funcs:
        funcs = [v for k, v in ns.items() if callable(v) and getattr(v, "__name__", "") != "<lambda>"]
    if not funcs:
        raise ValueError(f"No function found in pasted source for {field_name}.")
    if len(funcs) > 1:
        name = _prompt_required(f"  Multiple functions defined; enter the one to use for {field_name}: ")
        if name not in ns or not callable(ns[name]):
            raise ValueError(f"'{name}' is not a defined function.")
        return ns[name]
    return funcs[0]


def _collect_dict(label: str, value_kind: str, *, required: bool) -> dict:
    print(f"\n{label}")
    print(f"  Enter pairs one at a time. Leave the key blank to finish.")
    out: dict = {}
    while True:
        key = _prompt("  key: ")
        if not key:
            if required and not out:
                print("  (at least one entry is required)")
                continue
            break
        raw = _prompt(f"  {value_kind} for '{key}': ")
        if value_kind == "value (number)":
            try:
                out[key] = _parse_number(raw)
            except ValueError:
                print(f"  '{raw}' is not a number; try again.")
                continue
        else:
            out[key] = raw
    return out


def _collect_sweep_axis(axis: str) -> dict | None:
    ans = _prompt(f"  Configure {axis}-axis sweep? (y/skip): ").lower()
    if ans not in ("y", "yes"):
        return None
    param = _prompt_required(f"    {axis} param name: ")
    lo = _parse_number(_prompt_required(f"    {axis} min: "))
    hi = _parse_number(_prompt_required(f"    {axis} max: "))
    step = _parse_number(_prompt_required(f"    {axis} step: "))
    return {"param": param, "range": (float(lo), float(hi), float(step))}


def _register_walkthrough() -> None:
    print("\n--- Register a custom model ---")
    print("At any optional step, type 'skip' to omit it.\n")

    while True:
        name = _prompt_required("Model name (required, unique identifier, e.g. 'ssh'): ")
        if name in _MODELS:
            print(f"  a model named '{name}' is already registered; pick another.")
            continue
        break
    display_name = _prompt_required("Display name (required, human-readable, e.g. 'SSH'): ")

    default_params = _collect_dict(
        "default_params (optional): numeric defaults for the model's Hamiltonian parameters",
        "value (number)",
        required=False,
    )
    param_labels = _collect_dict(
        "param_labels (required): display labels for each parameter (include sweep params too)",
        "label (string)",
        required=True,
    )

    callables: dict[str, object] = {}
    source_blocks: dict[str, str] = {}

    for field_name, hint in [
        ("hamiltonian_matrix", "(n_sites, **params) -> np.ndarray"),
        ("fermionic_hamiltonian", "(n_sites, *, **params) -> FermionicOp"),
        ("get_optimizer", "(max_iters: int) -> Optimizer"),
        ("mean_field_correction", "(n_sites, n_occ, **params) -> float"),
    ]:
        src = _read_paste_block(field_name, hint)
        if src is None:
            continue
        try:
            fn = _exec_callable(src, field_name)
        except Exception as e:
            print(f"error compiling {field_name}: {e}")
            print("aborting registration.")
            return
        callables[field_name] = fn
        source_blocks[field_name] = src

    print("\nsweep_defaults (optional)")
    sweep_defaults: dict = {}
    x = _collect_sweep_axis("x")
    if x:
        sweep_defaults["x"] = x
    y = _collect_sweep_axis("y")
    if y:
        sweep_defaults["y"] = y

    print("\n--- Summary ---")
    print(f"  name:           {name}")
    print(f"  display_name:   {display_name}")
    print(f"  default_params: {default_params}")
    print(f"  param_labels:   {param_labels}")
    print(f"  callables:      {sorted(callables)}")
    print(f"  sweep_defaults: {sweep_defaults or '(none)'}")
    confirm = _prompt("\nWrite this model? (y/n): ").lower()
    if confirm not in ("y", "yes"):
        print("aborted; nothing written.")
        return

    try:
        model = Model(
            name=name,
            display_name=display_name,
            default_params=default_params,
            param_labels=param_labels,
            hamiltonian_matrix=callables.get("hamiltonian_matrix"),
            fermionic_hamiltonian=callables.get("fermionic_hamiltonian"),
            get_optimizer=callables.get("get_optimizer"),
            mean_field_correction=callables.get("mean_field_correction"),
            sweep_defaults=sweep_defaults or None,
        )
        register_model(model, _source_blocks=source_blocks)
    except Exception as e:
        print(f"error: {e}")
        return
    print(f"Registered '{name}'.")
