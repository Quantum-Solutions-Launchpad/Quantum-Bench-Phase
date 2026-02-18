from pathlib import Path
from utils import real_space_exact, setup_logging
import numpy as np
import matplotlib.pyplot as plt
import json

setup_logging()

n_sites = 4
t1, t2, phi = 1.0, 0.05, np.pi/4
spin = 2
max_n_occ = spin * n_sites

ROOT = Path(__file__).resolve().parents[2]
dmrg_json_path = ROOT / "cache" / "haldane-model" / "real-space" / "dmrg" / f"dmrg-ideal-t2-{t2}.json"

# load DMRG energies from json file
with dmrg_json_path.open("r") as f:
    j = json.load(f)

E_dmrg = np.array(j["energies"], dtype=float)  # length max_n_occ+1

# compute exact value to overlay
E_exact = np.array([real_space_exact(n_sites, t1, t2, phi, n_occ)
                    for n_occ in range(max_n_occ + 1)], dtype=float)

# plotting
plt.figure()
plt.plot(range(max_n_occ + 1), E_exact, "ro-", label="Exact")
plt.plot(range(max_n_occ + 1), E_dmrg, "ko", label="DMRG (Julia)")

plt.xlabel("Particle Number ($n_{occ}$)")
plt.ylabel("$E$")
plt.title(
    "Real-Space Haldane Hamiltonian Ground State Energy\n"
    f"$t_1={t1}, t_2={t2}, \\phi=\\pi/{int(np.pi/phi)}, N_{{sites}}={n_sites}$"
)
plt.legend()
plt.tight_layout()

outdir = ROOT / "plots" / "haldane-model" / "real-space" / f"{n_sites}-sites"
outdir.mkdir(parents=True, exist_ok=True)
file_path = outdir / f"dmrg-ideal-t2-{t2}.png"
plt.savefig(file_path, dpi=200)
plt.show()

print("Saved:", file_path)
