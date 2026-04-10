import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import eigsh


##--------------------##
##Outputfoldercreator##
##--------------------##
def create_run_directory(base_dir="output"):
    ##Createbaseoutputfolderifmissing##
    os.makedirs(base_dir, exist_ok=True)

    ##Timestampedsubfolder##
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


##--------------------##
##Utility:dotmask##
##--------------------##
def make_center_dot_mask(L: int, Ld: int) -> np.ndarray:
    ##CreatebooleanmaskTrueinsidedotsegment##
    if Ld <= 0 or Ld > L:
        raise ValueError("Ld must satisfy 1 <= Ld <= L")
    start = (L - Ld) // 2
    end = start + Ld
    mask = np.zeros(L, dtype=bool)
    mask[start:end] = True
    return mask


def make_mu_profile(mu_base: float, dot_mask: np.ndarray, V0: float) -> np.ndarray:
    ##Embedded-dotconfinement:raiseenergyoutsidedot##
    mu = np.full(dot_mask.size, mu_base, dtype=float)
    mu[~dot_mask] = mu_base + V0
    return mu


def dot_boundaries_from_mask(dot_mask: np.ndarray):
    ##Return(left_index,right_index)ofdotsegment##
    inds = np.where(dot_mask)[0]
    return int(inds.min()), int(inds.max())


##--------------------##
##BdGbuilder##
##--------------------##
def build_kitaev_bdg(L, t, mu_profile, Delta, dot_mask=None, g_boundary=None):
    ##NormalblockA##
    A = lil_matrix((L, L), dtype=np.complex128)
    for i in range(L):
        A[i, i] = -mu_profile[i]

    ##PairingblockB##
    B = lil_matrix((L, L), dtype=np.complex128)

    def link_scale(i, j):
        ##Scaleonlylinksthatcrossdotboundary##
        if g_boundary is None or dot_mask is None:
            return 1.0
        if bool(dot_mask[i]) != bool(dot_mask[j]):
            return float(g_boundary)
        return 1.0

    for i in range(L - 1):
        s = link_scale(i, i + 1)

        ##Hopping##
        A[i, i + 1] += -s * t
        A[i + 1, i] += -s * t

        ##p-wavepairing(antisymmetric)##
        B[i, i + 1] += s * Delta
        B[i + 1, i] += -s * Delta

    ##AssembleBdG##
    H = lil_matrix((2 * L, 2 * L), dtype=np.complex128)
    H[:L, :L] = A
    H[:L, L:] = B
    H[L:, :L] = -B.conjugate()
    H[L:, L:] = -A.transpose()

    return csc_matrix(H)


##--------------------##
##Eigen-solve##
##--------------------##
def solve_near_zero_modes(H, k: int = 8, sigma: float = 1e-9, max_tries: int = 3):
    """
    Robustnear-zeroeigensolver.
    1)shift-invertwithtinysigma(avoidsingularfactorization)
    2)fallbacktosmallest-magnitudeifneeded
    """
    ##Tryshift-invert##
    for _ in range(max_tries):
        try:
            evals, evecs = eigsh(H, k=k, sigma=sigma, which="LM")
            order = np.argsort(np.abs(evals))
            return evals[order], evecs[:, order]
        except RuntimeError:
            sigma *= 10.0

    ##Fallback##
    evals, evecs = eigsh(H, k=k, which="SM")
    order = np.argsort(np.abs(evals))
    return evals[order], evecs[:, order]


##--------------------##
##Metrics##
##--------------------##
def bdg_mode_density(evec, L):
    ##rho_i=|u_i|^2+|v_i|^2##
    u = evec[:L]
    v = evec[L:]
    rho = (np.abs(u) ** 2 + np.abs(v) ** 2).real
    s = rho.sum()
    return rho / s if s > 0 else rho


def dot_leakage(rho, dot_mask):
    return float(rho[dot_mask].sum())


def smallest_positive_energy(evals):
    pos = np.array([abs(e) for e in evals if e > 0])
    return float(pos.min()) if pos.size > 0 else np.nan


def end_weights(rho, n_end: int):
    ##Sumweightinleft/rightendwindows##
    left = float(rho[:n_end].sum())
    right = float(rho[-n_end:].sum())
    return left, right


@dataclass
class RunResult:
    V0: float
    Emin: float
    leakage_dot: float
    evals: np.ndarray


##--------------------##
##Sweep##
##--------------------##
def run_V0_sweep(L, Ld, t, Delta, mu_base, g_boundary, V0_values, k_modes):
    dot_mask = make_center_dot_mask(L, Ld)
    results = []

    for V0 in V0_values:
        mu_profile = make_mu_profile(mu_base, dot_mask, V0)
        H = build_kitaev_bdg(L, t, mu_profile, Delta, dot_mask, g_boundary)
        evals, evecs = solve_near_zero_modes(H, k=k_modes)

        psi0 = evecs[:, 0]
        rho0 = bdg_mode_density(psi0, L)

        Emin = smallest_positive_energy(evals)
        leak = dot_leakage(rho0, dot_mask)

        results.append(RunResult(float(V0), Emin, leak, evals))

    return results


##--------------------##
##Phase2:2Dsweep##
##--------------------##
def run_V0_g_sweep(L, Ld, t, Delta, mu_base, V0_values, g_values, k_modes):
    dot_mask = make_center_dot_mask(L, Ld)

    Emin_map = np.zeros((g_values.size, V0_values.size), dtype=float)
    leak_map = np.zeros((g_values.size, V0_values.size), dtype=float)

    for ig, g in enumerate(g_values):
        for iv, V0 in enumerate(V0_values):
            mu_profile = make_mu_profile(mu_base, dot_mask, V0)
            H = build_kitaev_bdg(
                L=L,
                t=t,
                mu_profile=mu_profile,
                Delta=Delta,
                dot_mask=dot_mask,
                g_boundary=float(g),
            )
            evals, evecs = solve_near_zero_modes(H, k=k_modes)

            psi0 = evecs[:, 0]
            rho0 = bdg_mode_density(psi0, L=L)

            Emin_map[ig, iv] = smallest_positive_energy(evals)
            leak_map[ig, iv] = dot_leakage(rho0, dot_mask)

    return Emin_map, leak_map


##--------------------##
##PlotFunctions:basic##
##--------------------##
def save_emin_plot(results, run_dir):
    V0 = np.array([r.V0 for r in results])
    Emin = np.array([r.Emin for r in results])

    plt.figure()
    plt.plot(V0, Emin, marker="o")
    plt.yscale("log")
    plt.xlabel("V0")
    plt.ylabel("Emin")
    plt.title("Emin vs V0")
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "emin_vs_V0.png"), dpi=200, bbox_inches="tight")
    plt.close()


def save_leakage_plot(results, run_dir):
    V0 = np.array([r.V0 for r in results])
    leak = np.array([r.leakage_dot for r in results])

    plt.figure()
    plt.plot(V0, leak, marker="o")
    plt.xlabel("V0")
    plt.ylabel("Dot Leakage")
    plt.title("Leakage vs V0")
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "leakage_vs_V0.png"), dpi=200, bbox_inches="tight")
    plt.close()


def save_heatmap(data, x, y, xlabel, ylabel, title, outpath, logscale=False):
    plt.figure()
    if logscale:
        ##Avoidlog10(0)##
        safe = np.maximum(data, 1e-20)
        plot_data = np.log10(safe)
        im = plt.imshow(
            plot_data,
            origin="lower",
            aspect="auto",
            extent=[x.min(), x.max(), y.min(), y.max()],
        )
        plt.colorbar(im, label="log10(value)")
    else:
        im = plt.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=[x.min(), x.max(), y.min(), y.max()],
        )
        plt.colorbar(im, label="value")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def save_spectrum_vs_V0(L, Ld, t, Delta, mu_base, g_fixed, V0_values, k_modes, n_levels, outpath):
    dot_mask = make_center_dot_mask(L, Ld)

    Epos = np.zeros((V0_values.size, n_levels), dtype=float)
    Epos[:] = np.nan

    for iv, V0 in enumerate(V0_values):
        mu_profile = make_mu_profile(mu_base, dot_mask, V0)
        H = build_kitaev_bdg(
            L=L,
            t=t,
            mu_profile=mu_profile,
            Delta=Delta,
            dot_mask=dot_mask,
            g_boundary=float(g_fixed),
        )
        evals, _ = solve_near_zero_modes(H, k=k_modes)
        pos = np.sort(evals[evals > 0.0])
        pos = pos[:n_levels] if pos.size >= n_levels else pos
        Epos[iv, :pos.size] = pos

    plt.figure()
    for k in range(n_levels):
        plt.plot(V0_values, Epos[:, k], marker="o", linestyle="-", label=f"level{k+1}")
    plt.yscale("log")
    plt.xlabel("V0")
    plt.ylabel("Positive near-zero energies")
    plt.title(f"Low-energy spectrum vs V0 (g={g_fixed:.2f})")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


##--------------------##
##PlotFunctions:new##
##--------------------##
def save_physical_setup_plot(run_dir, L, Ld, mu_base, V0, g_boundary):
    """
    Plotmu_profile(i)andannotatedotregionboundaries.
    Thisisyour"whatisphysicallyhappening"figure.
    """
    dot_mask = make_center_dot_mask(L, Ld)
    mu_profile = make_mu_profile(mu_base, dot_mask, V0)
    left_b, right_b = dot_boundaries_from_mask(dot_mask)
    x = np.arange(L)

    plt.figure()
    plt.plot(x, mu_profile, linewidth=2)
    plt.axvline(left_b, linestyle="--")
    plt.axvline(right_b, linestyle="--")

    ##Annotategboundaryatinterfaces##
    plt.text(left_b, mu_profile.max(), f" g={g_boundary:.2f} ", va="bottom", ha="left")
    plt.text(right_b, mu_profile.max(), f" g={g_boundary:.2f} ", va="bottom", ha="right")

    plt.xlabel("Site index i")
    plt.ylabel("mu(i)")
    plt.title(f"Physical setup: embedded dot via mu profile (V0={V0:.2f}, g={g_boundary:.2f})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(run_dir, f"setup_mu_profile_V0_{V0:.2f}_g_{g_boundary:.2f}.png"), dpi=200, bbox_inches="tight")
    plt.close()


def solve_and_get_lowest_modes(L, Ld, t, Delta, mu_base, V0, g_boundary, k_modes):
    ##Helper:getevalsanddensitiesforlow-energyBdGmodes##
    dot_mask = make_center_dot_mask(L, Ld)
    mu_profile = make_mu_profile(mu_base, dot_mask, V0)
    H = build_kitaev_bdg(L, t, mu_profile, Delta, dot_mask, g_boundary)
    evals, evecs = solve_near_zero_modes(H, k=k_modes)

    rhos = []
    for k in range(evecs.shape[1]):
        rhos.append(bdg_mode_density(evecs[:, k], L))
    rhos = np.array(rhos)  ##shape:(k_modes,L)##
    return evals, rhos, dot_mask


def save_zero_mode_end_plot(run_dir, L, Ld, t, Delta, mu_base, V0, g_boundary, k_modes, n_end=20):
    """
    Plotdensityrho_iofclosest-to-zeromode.
    Thisisyour"zeromodesattheends"figure.
    """
    evals, rhos, dot_mask = solve_and_get_lowest_modes(L, Ld, t, Delta, mu_base, V0, g_boundary, k_modes)
    rho0 = rhos[0]
    left_b, right_b = dot_boundaries_from_mask(dot_mask)
    x = np.arange(L)

    left_w, right_w = end_weights(rho0, n_end)

    plt.figure()
    plt.plot(x, rho0, linewidth=2)
    plt.axvline(left_b, linestyle="--")
    plt.axvline(right_b, linestyle="--")

    ##Markendwindows##
    plt.axvspan(0, n_end - 1, alpha=0.15)
    plt.axvspan(L - n_end, L - 1, alpha=0.15)

    plt.xlabel("Site index i")
    plt.ylabel("rho_i (normalized)")
    plt.title(
        f"Closest-to-zero mode density |E0|={abs(evals[0]):.2e}\n"
        f"End weights: left={left_w:.3f}, right={right_w:.3f}  (V0={V0:.2f}, g={g_boundary:.2f})"
    )
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(run_dir, f"zero_mode_density_V0_{V0:.2f}_g_{g_boundary:.2f}.png"), dpi=200, bbox_inches="tight")
    plt.close()


def save_compare_spectra_before_after(run_dir, L, Ld, t, Delta, mu_base, g_boundary, k_modes, V0_before, V0_after):
    """
    Comparelow-energy|E|valuesbefore/afterdotinduction.
    Thisletscolleague"seewherezeromodesappear"numerically.
    """
    evals_b, _, _ = solve_and_get_lowest_modes(L, Ld, t, Delta, mu_base, V0_before, g_boundary, k_modes)
    evals_a, _, _ = solve_and_get_lowest_modes(L, Ld, t, Delta, mu_base, V0_after, g_boundary, k_modes)

    Eb = np.sort(np.abs(evals_b))
    Ea = np.sort(np.abs(evals_a))
    nshow = min(10, Eb.size, Ea.size)

    plt.figure()
    plt.plot(np.arange(nshow), Eb[:nshow], marker="o", linestyle="-", label=f"before V0={V0_before:.2f}")
    plt.plot(np.arange(nshow), Ea[:nshow], marker="o", linestyle="-", label=f"after V0={V0_after:.2f}")
    plt.yscale("log")
    plt.xlabel("Mode index (sorted by |E|)")
    plt.ylabel("|E|")
    plt.title(f"Low-energy spectrum magnitude before vs after dot induction (g={g_boundary:.2f})")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(os.path.join(run_dir, f"compare_spectra_V0_{V0_before:.2f}_to_{V0_after:.2f}_g_{g_boundary:.2f}.png"), dpi=200, bbox_inches="tight")
    plt.close()


def save_multiple_mode_densities(run_dir, L, Ld, t, Delta, mu_base, V0, g_boundary, k_modes, n_plot=4):
    """
    Plotdensitiesofseverallow-energyBdGmodes(nearto0).
    Usefulwhenmultiplezeromodesappear(domain-wallmodes).
    """
    evals, rhos, dot_mask = solve_and_get_lowest_modes(L, Ld, t, Delta, mu_base, V0, g_boundary, k_modes)
    left_b, right_b = dot_boundaries_from_mask(dot_mask)
    x = np.arange(L)

    plt.figure()
    for k in range(min(n_plot, rhos.shape[0])):
        plt.plot(x, rhos[k], label=f"k={k}, E={evals[k]:+.2e}")

    plt.axvline(left_b, linestyle="--")
    plt.axvline(right_b, linestyle="--")
    plt.xlabel("Site index i")
    plt.ylabel("rho_i (normalized)")
    plt.title(f"Several lowest |E| mode densities (V0={V0:.2f}, g={g_boundary:.2f})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(os.path.join(run_dir, f"multi_mode_densities_V0_{V0:.2f}_g_{g_boundary:.2f}.png"), dpi=200, bbox_inches="tight")
    plt.close()


##--------------------##
##Main##
##--------------------##
if __name__ == "__main__":

    ##Parameters##
    L = 200
    Ld = 40
    t = 1.0
    Delta = 0.3
    mu_base = 0.5
    g_boundary = 1.0
    k_modes = 12
    V0_values = np.linspace(0.0, 8.0, 41)

    ##Createoutputdirectory##
    run_dir = create_run_directory()
    print(f"Saving results to: {run_dir}")

    ##Run V0 sweep##
    results = run_V0_sweep(L, Ld, t, Delta, mu_base, g_boundary, V0_values, k_modes)

    ##Save basic plots##
    save_emin_plot(results, run_dir)
    save_leakage_plot(results, run_dir)

    ##Save data##
    np.savez(
        os.path.join(run_dir, "data_V0_sweep.npz"),
        V0=np.array([r.V0 for r in results]),
        Emin=np.array([r.Emin for r in results]),
        leakage=np.array([r.leakage_dot for r in results]),
    )

    ##Save parameter info##
    with open(os.path.join(run_dir, "params.txt"), "w") as f:
        f.write(f"L = {L}\n")
        f.write(f"Ld = {Ld}\n")
        f.write(f"t = {t}\n")
        f.write(f"Delta = {Delta}\n")
        f.write(f"mu_base = {mu_base}\n")
        f.write(f"g_boundary = {g_boundary}\n")
        f.write(f"k_modes = {k_modes}\n")

    ##------------------------------##
    ##NEW:colleague-friendlyfigures##
    ##------------------------------##
    ##Baseline(noinduceddot)##
    V0_before = 0.0

    ##Induceddot/boundaryexample##
    V0_after = 6.0

    ##Physicalsetupplots##
    save_physical_setup_plot(run_dir, L, Ld, mu_base, V0_before, g_boundary)
    save_physical_setup_plot(run_dir, L, Ld, mu_base, V0_after, g_boundary)

    ##Zero-modeatends(baseline)andafterinduction##
    save_zero_mode_end_plot(run_dir, L, Ld, t, Delta, mu_base, V0_before, g_boundary, k_modes, n_end=20)
    save_zero_mode_end_plot(run_dir, L, Ld, t, Delta, mu_base, V0_after, g_boundary, k_modes, n_end=20)

    ##Comparelow-energyenergiesbefore/after##
    save_compare_spectra_before_after(run_dir, L, Ld, t, Delta, mu_base, g_boundary, k_modes, V0_before, V0_after)

    ##Showseverallow-energydensities(afterinduction)##
    save_multiple_mode_densities(run_dir, L, Ld, t, Delta, mu_base, V0_after, g_boundary, k_modes, n_plot=4)

    print("Phase1+colleaguefigures saved.")

    ##--------------------##
    ##Phase2:(V0,g)sweep##
    ##--------------------##
    ##Avoidexactg=0forstability;youcanrung=0separatelylater##
    g_values = np.linspace(0.02, 1.0, 26)

    Emin_map, leak_map = run_V0_g_sweep(
        L=L,
        Ld=Ld,
        t=t,
        Delta=Delta,
        mu_base=mu_base,
        V0_values=V0_values,
        g_values=g_values,
        k_modes=k_modes,
    )

    save_heatmap(
        data=Emin_map,
        x=V0_values,
        y=g_values,
        xlabel="V0",
        ylabel="g_boundary",
        title="Emin(V0,g)",
        outpath=os.path.join(run_dir, "heatmap_Emin.png"),
        logscale=True,
    )

    save_heatmap(
        data=leak_map,
        x=V0_values,
        y=g_values,
        xlabel="V0",
        ylabel="g_boundary",
        title="Dot leakage(V0,g)",
        outpath=os.path.join(run_dir, "heatmap_leakage.png"),
        logscale=False,
    )

    np.savez(
        os.path.join(run_dir, "data_V0_g_sweep.npz"),
        V0=V0_values,
        g=g_values,
        Emin_map=Emin_map,
        leak_map=leak_map,
    )

    ##Low-energyspectrumvsV0atg=1##
    save_spectrum_vs_V0(
        L=L,
        Ld=Ld,
        t=t,
        Delta=Delta,
        mu_base=mu_base,
        g_fixed=1.0,
        V0_values=V0_values,
        k_modes=k_modes,
        n_levels=6,
        outpath=os.path.join(run_dir, "spectrum_vs_V0_g1.png"),
    )

    print("Phase2 heatmaps and spectrum saved.")