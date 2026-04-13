import os
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime


##--------------------##
##Outputfoldercreator##
##--------------------##
def create_run_directory(base_dir="output"):
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


##------------------##
##Geometryconstants##
##------------------##
SQRT3 = np.sqrt(3.0)

##HoneycombBravaisvectors##
a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, SQRT3 / 2.0])

##Basisoffset:A->B##
d1 = (a1 + a2) / 3.0

##NNvectorsfromAtoB##
NN_DISPS = [
    d1,
    d1 - a1,
    d1 - a2,
]

##NNNvectorsonsamesublattice##
NNN_POSITIVE = [
    a1,
    a2,
    a2 - a1,
]

NNN_ALL = [
    a1,
    a2,
    a2 - a1,
    -a1,
    -a2,
    -(a2 - a1),
]


##----------------##
##Utilityhelpers##
##----------------##
def round_key(vec, ndigits=10):
    return tuple(np.round(vec, ndigits))


def complex_json_safe(val):
    if np.iscomplexobj(val):
        return {"real": float(np.real(val)), "imag": float(np.imag(val))}
    return float(val)


##----------------##
##Sitedataclass##
##----------------##
@dataclass
class Site:
    idx: int
    cell_n1: int
    cell_n2: int
    sublattice: int  ##+1forA,-1forB##
    x: float
    y: float


##---------------------##
##Honeycombdotbuilder##
##---------------------##
def build_disk_honeycomb_dot(R, pad=3):
    """
    BuildalargerectangularhoneycombpatchandkeeponlysitesinsidediskradiusR.
    """
    sites = []
    idx = 0

    ##Estimatehowmanyunitcellsweneedaroundtheorigin##
    nmax = int(np.ceil(R + pad))

    for n1 in range(-nmax, nmax + 1):
        for n2 in range(-nmax, nmax + 1):
            rA = n1 * a1 + n2 * a2
            rB = rA + d1

            if np.linalg.norm(rA) <= R:
                sites.append(Site(idx=idx, cell_n1=n1, cell_n2=n2, sublattice=+1, x=rA[0], y=rA[1]))
                idx += 1

            if np.linalg.norm(rB) <= R:
                sites.append(Site(idx=idx, cell_n1=n1, cell_n2=n2, sublattice=-1, x=rB[0], y=rB[1]))
                idx += 1

    return sites


def build_position_lookup(sites, ndigits=10):
    pos_to_idx = {}
    for s in sites:
        pos_to_idx[round_key(np.array([s.x, s.y]), ndigits=ndigits)] = s.idx
    return pos_to_idx


##---------------------##
##Neighborconstruction##
##---------------------##
def find_nn_pairs(sites, ndigits=10):
    """
    ReturnundirectedNNpairs(i,j)withi<j.
    """
    pos_to_idx = build_position_lookup(sites, ndigits=ndigits)
    pairs = set()

    for s in sites:
        if s.sublattice != +1:
            continue

        rA = np.array([s.x, s.y])
        i = s.idx

        for disp in NN_DISPS:
            rB = rA + disp
            key = round_key(rB, ndigits=ndigits)
            if key in pos_to_idx:
                j = pos_to_idx[key]
                pair = tuple(sorted((i, j)))
                pairs.add(pair)

    pairs = sorted(list(pairs))
    return pairs


def nnn_direction_sign(delta, tol=1e-9):
    """
    Return+1forpositiveNNNdirections,-1fornegativeones.
    """
    for vec in NNN_POSITIVE:
        if np.linalg.norm(delta - vec) < tol:
            return +1
        if np.linalg.norm(delta + vec) < tol:
            return -1
    raise ValueError(f"Could not classify NNN displacement delta={delta}")


def find_nnn_pairs_with_nu(sites, ndigits=10, tol=1e-9):
    """
    ReturndirectedNNNhops(i,j,nu)onlyonceforeachundirectedpairwithi<j.
    Thephaseconventionusedhereis:
    - A-sublatticepositiveNNNdirections:+phi
    - B-sublatticepositiveNNNdirections:-phi
    ReverseeddirectionscarrytheoppositephasebyHermitianconjugation.
    """
    pos_to_idx = build_position_lookup(sites, ndigits=ndigits)
    directed_pairs = []

    for s in sites:
        r0 = np.array([s.x, s.y])
        i = s.idx
        sub_sign = +1 if s.sublattice == +1 else -1

        for disp in NNN_ALL:
            r1 = r0 + disp
            key = round_key(r1, ndigits=ndigits)
            if key not in pos_to_idx:
                continue

            j = pos_to_idx[key]
            if i >= j:
                continue

            direction_sign = nnn_direction_sign(disp, tol=tol)
            nu = sub_sign * direction_sign
            directed_pairs.append((i, j, nu))

    return directed_pairs


##--------------------##
##Hamiltonianbuilder##
##--------------------##
def build_haldane_hamiltonian(sites, nn_pairs, nnn_pairs_nu, t, t2, phi, M):
    N = len(sites)
    H = np.zeros((N, N), dtype=np.complex128)

    ##Onsitemass##
    for s in sites:
        H[s.idx, s.idx] = M * s.sublattice

    ##NNhopping##
    for i, j in nn_pairs:
        H[i, j] += -t
        H[j, i] += -t

    ##NNNcomplexHaldanehopping##
    for i, j, nu in nnn_pairs_nu:
        amp = t2 * np.exp(1j * nu * phi)
        H[i, j] += amp
        H[j, i] += np.conjugate(amp)

    return H


def hermiticity_error(H):
    return float(np.max(np.abs(H - H.conjugate().T)))


##----------------##
##Eigenspectrum##
##----------------##
def solve_full_spectrum(H):
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs


##----------------##
##Stateanalysis##
##----------------##
def get_positions_array(sites):
    xy = np.zeros((len(sites), 2), dtype=float)
    sub = np.zeros(len(sites), dtype=int)
    for s in sites:
        xy[s.idx, 0] = s.x
        xy[s.idx, 1] = s.y
        sub[s.idx] = s.sublattice
    return xy, sub


def state_density(psi):
    rho = np.abs(psi) ** 2
    s = rho.sum()
    return rho / s if s > 0 else rho


def radial_distances(sites):
    xy, _ = get_positions_array(sites)
    return np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)


def make_edge_mask(sites, edge_width):
    r = radial_distances(sites)
    rmax = r.max()
    mask = r >= (rmax - edge_width)
    return mask


def edge_participation(evecs, edge_mask):
    parts = np.zeros(evecs.shape[1], dtype=float)
    for k in range(evecs.shape[1]):
        rho = state_density(evecs[:, k])
        parts[k] = float(rho[edge_mask].sum())
    return parts


def bulk_participation(evecs, bulk_mask):
    parts = np.zeros(evecs.shape[1], dtype=float)
    for k in range(evecs.shape[1]):
        rho = state_density(evecs[:, k])
        parts[k] = float(rho[bulk_mask].sum())
    return parts


def pick_edge_like_near_gap_state(evals, evecs, edge_parts, n_candidates=20):
    """
    Fromtheclosest-to-zerostates,picktheonewithlargestedgeparticipation.
    """
    order = np.argsort(np.abs(evals))
    cand = order[:min(n_candidates, len(order))]
    best = cand[np.argmax(edge_parts[cand])]
    return int(best)


##----------------##
##Bondcurrents##
##----------------##
def bond_current_for_state(H, psi, i, j):
    return float(-2.0 * np.imag(np.conjugate(psi[i]) * H[i, j] * psi[j]))


def compute_currents_for_bonds(H, psi, bonds):
    vals = []
    for i, j in bonds:
        J = bond_current_for_state(H, psi, i, j)
        vals.append((i, j, J))
    return vals


##----------------##
##Plotfunctions##
##----------------##
def save_lattice_sites_plot(run_dir, sites, R):
    xy, sub = get_positions_array(sites)

    plt.figure(figsize=(7, 7))
    maskA = sub == +1
    maskB = sub == -1

    plt.scatter(xy[maskA, 0], xy[maskA, 1], s=18, label="A")
    plt.scatter(xy[maskB, 0], xy[maskB, 1], s=18, label="B")

    circle = plt.Circle((0.0, 0.0), R, fill=False, linestyle="--")
    plt.gca().add_artist(circle)

    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Finite Haldane dot lattice sites")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(run_dir, "lattice_sites.png"), dpi=220, bbox_inches="tight")
    plt.close()


def save_geometry_bonds_plot(run_dir, sites, nn_pairs):
    xy, sub = get_positions_array(sites)

    plt.figure(figsize=(7, 7))

    for i, j in nn_pairs:
        plt.plot(
            [xy[i, 0], xy[j, 0]],
            [xy[i, 1], xy[j, 1]],
            linewidth=0.6,
            alpha=0.6,
        )

    maskA = sub == +1
    maskB = sub == -1
    plt.scatter(xy[maskA, 0], xy[maskA, 1], s=12, label="A")
    plt.scatter(xy[maskB, 0], xy[maskB, 1], s=12, label="B")

    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Geometry sanity check: lattice + NN bonds")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(run_dir, "geometry_nn_bonds.png"), dpi=220, bbox_inches="tight")
    plt.close()


def save_spectrum_plot(run_dir, evals):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(evals)), evals, marker="o", linestyle="None", markersize=3)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Eigenstate index")
    plt.ylabel("Energy")
    plt.title("Full spectrum of finite Haldane dot")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(run_dir, "spectrum_full.png"), dpi=220, bbox_inches="tight")
    plt.close()


def save_zoomed_spectrum_plot(run_dir, evals, edge_parts, n_show=60):
    order = np.argsort(np.abs(evals))[:min(n_show, len(evals))]
    x = np.arange(order.size)

    plt.figure(figsize=(8, 5))
    sc = plt.scatter(x, evals[order], c=edge_parts[order], s=35)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("State rank by |E|")
    plt.ylabel("Energy")
    plt.title("Near-gap spectrum colored by edge participation")
    cbar = plt.colorbar(sc)
    cbar.set_label("Edge participation")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(run_dir, "spectrum_near_gap_colored.png"), dpi=220, bbox_inches="tight")
    plt.close()


def save_edge_participation_plot(run_dir, evals, edge_parts):
    plt.figure(figsize=(8, 5))
    plt.scatter(np.arange(len(evals)), edge_parts, s=12)
    plt.xlabel("Eigenstate index")
    plt.ylabel("Edge participation")
    plt.title("Edge participation vs eigenstate index")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(run_dir, "edge_participation_vs_index.png"), dpi=220, bbox_inches="tight")
    plt.close()

    order = np.argsort(np.abs(evals))
    plt.figure(figsize=(8, 5))
    plt.scatter(np.arange(len(evals)), edge_parts[order], s=12)
    plt.xlabel("State rank by |E|")
    plt.ylabel("Edge participation")
    plt.title("Edge participation vs rank in |E|")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(run_dir, "edge_participation_vs_absE_rank.png"), dpi=220, bbox_inches="tight")
    plt.close()


def save_selected_state_density_plot(run_dir, sites, evals, evecs, state_index, edge_mask):
    xy, _ = get_positions_array(sites)
    psi = evecs[:, state_index]
    rho = state_density(psi)

    plt.figure(figsize=(7, 7))
    sc = plt.scatter(xy[:, 0], xy[:, 1], c=rho, s=35 + 300 * rho, edgecolors="none")
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        f"Selected state density\n"
        f"state={state_index}, E={evals[state_index]:+.6f}, edge_weight={rho[edge_mask].sum():.3f}"
    )
    cbar = plt.colorbar(sc)
    cbar.set_label(r"$|\psi_i|^2$")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(run_dir, "selected_state_density.png"), dpi=220, bbox_inches="tight")
    plt.close()


def save_selected_state_density_with_boundary_plot(run_dir, sites, evals, evecs, state_index, edge_width):
    xy, _ = get_positions_array(sites)
    psi = evecs[:, state_index]
    rho = state_density(psi)
    r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    rmax = r.max()

    plt.figure(figsize=(7, 7))
    sc = plt.scatter(xy[:, 0], xy[:, 1], c=rho, s=35 + 300 * rho, edgecolors="none")

    outer = plt.Circle((0.0, 0.0), rmax, fill=False, linestyle="--")
    inner = plt.Circle((0.0, 0.0), max(rmax - edge_width, 0.0), fill=False, linestyle=":")
    plt.gca().add_artist(outer)
    plt.gca().add_artist(inner)

    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Selected state density with edge shell\nE={evals[state_index]:+.6f}")
    cbar = plt.colorbar(sc)
    cbar.set_label(r"$|\psi_i|^2$")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(run_dir, "selected_state_density_edge_shell.png"), dpi=220, bbox_inches="tight")
    plt.close()


def save_current_plot(run_dir, sites, currents, title, filename, current_threshold_fraction=0.08):
    xy, _ = get_positions_array(sites)

    absJ = np.array([abs(J) for _, _, J in currents], dtype=float)
    if absJ.size == 0 or absJ.max() == 0.0:
        plt.figure(figsize=(7, 7))
        plt.scatter(xy[:, 0], xy[:, 1], s=12)
        plt.gca().set_aspect("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title + "\n(no visible current)")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.savefig(os.path.join(run_dir, filename), dpi=220, bbox_inches="tight")
        plt.close()
        return

    cutoff = current_threshold_fraction * absJ.max()

    plt.figure(figsize=(7, 7))
    plt.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.35)

    for i, j, J in currents:
        if abs(J) < cutoff:
            continue

        x0, y0 = xy[i]
        x1, y1 = xy[j]

        color = "red" if J > 0 else "blue"
        lw = 0.5 + 3.5 * abs(J) / absJ.max()

        plt.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=0.85)

    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(run_dir, filename), dpi=220, bbox_inches="tight")
    plt.close()


def save_radial_profile_plot(run_dir, sites, evals, evecs, state_index):
    xy, _ = get_positions_array(sites)
    psi = evecs[:, state_index]
    rho = state_density(psi)
    r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)

    order = np.argsort(r)

    plt.figure(figsize=(8, 5))
    plt.plot(r[order], rho[order], marker="o", linestyle="None", markersize=3)
    plt.xlabel("Radial distance from center")
    plt.ylabel(r"$|\psi_i|^2$")
    plt.title(f"Radial profile of selected state\nE={evals[state_index]:+.6f}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(run_dir, "selected_state_radial_profile.png"), dpi=220, bbox_inches="tight")
    plt.close()


##--------------------##
##Dataexporthelpers##
##--------------------##
def site_table_array(sites):
    arr = np.zeros((len(sites), 6), dtype=float)
    for s in sites:
        arr[s.idx, 0] = s.idx
        arr[s.idx, 1] = s.cell_n1
        arr[s.idx, 2] = s.cell_n2
        arr[s.idx, 3] = s.sublattice
        arr[s.idx, 4] = s.x
        arr[s.idx, 5] = s.y
    return arr


##------------##
##Mainrunner##
##------------##
if __name__ == "__main__":

    ##----------------##
    ##Phase1parameters##
    ##----------------##
    R = 8.0
    edge_width = 1.25

    t = 1.0
    t2 = 0.1
    phi = np.pi / 2.0
    M = 0.2

    ##----------------------##
    ##Createoutputdirectory##
    ##----------------------##
    run_dir = create_run_directory()
    print(f"Saving Phase 1 results to: {run_dir}")

    ##----------------##
    ##Buildgeometry##
    ##----------------##
    sites = build_disk_honeycomb_dot(R=R, pad=4)
    N = len(sites)

    print(f"Number of sites in finite dot: {N}")

    nn_pairs = find_nn_pairs(sites)
    nnn_pairs_nu = find_nnn_pairs_with_nu(sites)

    print(f"Number of NN bonds: {len(nn_pairs)}")
    print(f"Number of NNN bonds: {len(nnn_pairs_nu)}")

    ##------------------##
    ##BuildHamiltonian##
    ##------------------##
    H = build_haldane_hamiltonian(
        sites=sites,
        nn_pairs=nn_pairs,
        nnn_pairs_nu=nnn_pairs_nu,
        t=t,
        t2=t2,
        phi=phi,
        M=M,
    )

    herm_err = hermiticity_error(H)
    print(f"Hermiticity max error: {herm_err:.3e}")

    ##----------------##
    ##Solvefullsystem##
    ##----------------##
    evals, evecs = solve_full_spectrum(H)

    ##----------------##
    ##Stateanalysis##
    ##----------------##
    edge_mask = make_edge_mask(sites, edge_width=edge_width)
    r = radial_distances(sites)
    bulk_mask = r <= max(r.max() - 2.0 * edge_width, 0.0)

    edge_parts = edge_participation(evecs, edge_mask)
    bulk_parts = bulk_participation(evecs, bulk_mask)

    selected_state = pick_edge_like_near_gap_state(
        evals=evals,
        evecs=evecs,
        edge_parts=edge_parts,
        n_candidates=20,
    )

    psi_sel = evecs[:, selected_state]

    ##NNcurrents##
    nn_currents = compute_currents_for_bonds(H, psi_sel, nn_pairs)

    ##NNNcurrents##
    nnn_bonds_plain = [(i, j) for i, j, _ in nnn_pairs_nu]
    nnn_currents = compute_currents_for_bonds(H, psi_sel, nnn_bonds_plain)

    print(f"Selected state index: {selected_state}")
    print(f"Selected state energy: {evals[selected_state]:+.8f}")
    print(f"Selected state edge participation: {edge_parts[selected_state]:.4f}")
    print(f"Selected state bulk participation: {bulk_parts[selected_state]:.4f}")

    ##----------------##
    ##Saveplots##
    ##----------------##
    save_lattice_sites_plot(run_dir, sites, R)
    save_geometry_bonds_plot(run_dir, sites, nn_pairs)
    save_spectrum_plot(run_dir, evals)
    save_zoomed_spectrum_plot(run_dir, evals, edge_parts, n_show=80)
    save_edge_participation_plot(run_dir, evals, edge_parts)
    save_selected_state_density_plot(run_dir, sites, evals, evecs, selected_state, edge_mask)
    save_selected_state_density_with_boundary_plot(run_dir, sites, evals, evecs, selected_state, edge_width)
    save_radial_profile_plot(run_dir, sites, evals, evecs, selected_state)

    save_current_plot(
        run_dir=run_dir,
        sites=sites,
        currents=nn_currents,
        title=f"NN bond currents for selected state\nstate={selected_state}, E={evals[selected_state]:+.6f}",
        filename="selected_state_nn_currents.png",
        current_threshold_fraction=0.10,
    )

    save_current_plot(
        run_dir=run_dir,
        sites=sites,
        currents=nnn_currents,
        title=f"NNN bond currents for selected state\nstate={selected_state}, E={evals[selected_state]:+.6f}",
        filename="selected_state_nnn_currents.png",
        current_threshold_fraction=0.10,
    )

    ##----------------##
    ##Savedatafiles##
    ##----------------##
    np.savez(
        os.path.join(run_dir, "phase1_haldane_dot_data.npz"),
        evals=evals,
        evecs=evecs,
        edge_mask=edge_mask.astype(int),
        bulk_mask=bulk_mask.astype(int),
        edge_participation=edge_parts,
        bulk_participation=bulk_parts,
        selected_state=selected_state,
        site_table=site_table_array(sites),
        nn_pairs=np.array(nn_pairs, dtype=int),
        nnn_pairs_nu=np.array(nnn_pairs_nu, dtype=float),
    )

    params = {
        "R": R,
        "edge_width": edge_width,
        "t": t,
        "t2": t2,
        "phi": complex_json_safe(phi),
        "M": M,
        "N_sites": N,
        "N_nn_bonds": len(nn_pairs),
        "N_nnn_bonds": len(nnn_pairs_nu),
        "hermiticity_error": herm_err,
        "selected_state": selected_state,
        "selected_energy": complex_json_safe(evals[selected_state]),
        "selected_edge_participation": edge_parts[selected_state],
        "selected_bulk_participation": bulk_parts[selected_state],
    }

    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write("Phase 1: finite Haldane quantum dot baseline\n")
        f.write(f"N_sites = {N}\n")
        f.write(f"N_NN_bonds = {len(nn_pairs)}\n")
        f.write(f"N_NNN_bonds = {len(nnn_pairs_nu)}\n")
        f.write(f"Hermiticity max error = {herm_err:.6e}\n")
        f.write(f"Selected state index = {selected_state}\n")
        f.write(f"Selected state energy = {evals[selected_state]:+.10f}\n")
        f.write(f"Selected state edge participation = {edge_parts[selected_state]:.6f}\n")
        f.write(f"Selected state bulk participation = {bulk_parts[selected_state]:.6f}\n")

    print("Phase 1 Haldane dot run complete.")