import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


def plot_chain_dot_boundaries(
    N,
    dot_span=None,
    g_boundaries=None,
    V=None,
    g_profile=None,
    H=None,
    save_dir="output"
):

    ##Create timestamped output folder##
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    x = np.arange(N)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    ax0, ax1, ax2, ax3 = axes

    ##Chain schematic##
    ax0.plot(x, np.zeros_like(x), marker='o', linestyle='-', linewidth=1)
    ax0.set_ylabel("schematic")
    ax0.set_yticks([])

    if dot_span is not None:
        i0, i1 = dot_span
        ax0.axvspan(i0 - 0.5, i1 + 0.5, alpha=0.2)
        ax0.text((i0 + i1) / 2, 0.05, "dot region", ha="center", va="bottom")

    if g_boundaries is not None:
        for gb in g_boundaries:
            ax0.axvline(gb, linestyle="--", linewidth=1)
            ax0.text(gb, -0.05, "boundary", rotation=90, va="top", ha="right")

    ax0.set_title("1D Chain: Dot Region and Boundaries (Real Space)")

    ##Onsite potential profile##
    if V is not None:
        ax1.plot(x, V, marker='o', linewidth=1)
        ax1.set_ylabel("V_i")
        ax1.set_title("Onsite Potential Profile")
    else:
        ax1.set_ylabel("V_i")

    ##Coupling / g profile##
    if g_profile is not None:
        g_profile = np.asarray(g_profile)

        if g_profile.shape[0] == N:
            ax2.plot(x, g_profile, marker='o', linewidth=1)
            ax2.set_ylabel("g_i")

        elif g_profile.shape[0] == N - 1:
            ax2.plot(x[:-1] + 0.5, g_profile, marker='o', linewidth=1)
            ax2.set_ylabel("g_{i,i+1}")

        ax2.set_title("Coupling / g Profile")
    else:
        ax2.set_ylabel("g")

    ##Near-zero mode and energy density##
    if H is not None:

        H = np.asarray(H)
        evals, evecs = np.linalg.eigh(H)

        ##Find eigenstate closest to zero##
        iz = np.argmin(np.abs(evals))
        ez = evals[iz]
        psi = evecs[:, iz]
        weight = np.abs(psi)**2

        ax3.plot(x, weight, marker='o', linewidth=1, label=f"|psi|^2 (E ≈ {ez:.3e})")
        ax3.set_ylabel("weight")
        ax3.set_title("Near-Zero Mode Localization")

        ##Energy density per site##
        Hpsi = H @ psi
        e_i = np.real(np.conj(psi) * Hpsi)

        ax3b = ax3.twinx()
        ax3b.plot(x, e_i, linestyle='--', linewidth=1, label="energy density")
        ax3b.set_ylabel("e_i")

        ax3.legend(loc="upper left")
        ax3b.legend(loc="upper right")

    else:
        ax3.set_ylabel("weight / e_i")

    ax3.set_xlabel("site index i")

    plt.tight_layout()

    ##Save figure##
    fig_path = os.path.join(run_dir, "dot_boundary_visualization.png")
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"Saved results to: {run_dir}")
    print(f"Figure saved as: {fig_path}")

    return run_dir



##Example usage##
if __name__ == "__main__":

    N = 80
    dot_span = (28, 40)
    g_boundaries = [28, 41]

    ##Onsite potential##
    V = np.zeros(N)
    V[dot_span[0]:dot_span[1]+1] = 1.5

    ##Coupling profile##
    g_profile = np.ones(N-1)
    g_profile[dot_span[0]:dot_span[1]] = 0.2

    ##Tight-binding Hamiltonian##
    t = 1.0
    H = np.zeros((N, N))
    np.fill_diagonal(H, V)

    for i in range(N-1):
        H[i, i+1] = -t * g_profile[i]
        H[i+1, i] = -t * g_profile[i]

    plot_chain_dot_boundaries(
        N,
        dot_span,
        g_boundaries,
        V=V,
        g_profile=g_profile,
        H=H
    )