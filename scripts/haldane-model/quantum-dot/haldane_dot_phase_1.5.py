import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


##------------------##
##Geometryconstants##
##------------------##
SQRT3 = np.sqrt(3.0)

a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, SQRT3 / 2.0])
d1 = (a1 + a2) / 3.0

NN_DISPS = [
    d1,
    d1 - a1,
    d1 - a2,
]

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


##-------------##
##Sitedataclass##
##-------------##
@dataclass
class Site:
    idx: int
    cell_n1: int
    cell_n2: int
    sublattice: int
    x: float
    y: float


##---------------##
##Utilityhelpers##
##---------------##
def round_key(vec, ndigits=10):
    return tuple(np.round(vec, ndigits))


def get_positions_array(sites):
    xy = np.zeros((len(sites), 2), dtype=float)
    sub = np.zeros(len(sites), dtype=int)
    for s in sites:
        xy[s.idx, 0] = s.x
        xy[s.idx, 1] = s.y
        sub[s.idx] = s.sublattice
    return xy, sub


def radial_distances(sites):
    xy, _ = get_positions_array(sites)
    return np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)


def state_density(psi):
    rho = np.abs(psi) ** 2
    s = rho.sum()
    return rho / s if s > 0 else rho


##---------------------##
##Honeycombdotbuilder##
##---------------------##
def build_disk_honeycomb_dot(R, pad=3):
    sites = []
    idx = 0
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
                pairs.add(tuple(sorted((i, j))))

    return sorted(list(pairs))


def nnn_direction_sign(delta, tol=1e-9):
    for vec in NNN_POSITIVE:
        if np.linalg.norm(delta - vec) < tol:
            return +1
        if np.linalg.norm(delta + vec) < tol:
            return -1
    raise ValueError(f"Could not classify NNN displacement delta={delta}")


def find_nnn_pairs_with_nu(sites, ndigits=10, tol=1e-9):
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

    ##NNNcomplexhopping##
    for i, j, nu in nnn_pairs_nu:
        amp = t2 * np.exp(1j * nu * phi)
        H[i, j] += amp
        H[j, i] += np.conjugate(amp)

    return H


def hermiticity_error(H):
    return float(np.max(np.abs(H - H.conjugate().T)))


##----------------##
##Stateanalysis##
##----------------##
def make_edge_mask(sites, edge_width):
    r = radial_distances(sites)
    rmax = r.max()
    return r >= (rmax - edge_width)


def make_bulk_mask(sites, edge_width):
    r = radial_distances(sites)
    rmax = r.max()
    return r <= max(rmax - 2.0 * edge_width, 0.0)


def edge_participation_all(evecs, edge_mask):
    vals = np.zeros(evecs.shape[1], dtype=float)
    for k in range(evecs.shape[1]):
        rho = state_density(evecs[:, k])
        vals[k] = float(rho[edge_mask].sum())
    return vals


def bulk_participation_all(evecs, bulk_mask):
    vals = np.zeros(evecs.shape[1], dtype=float)
    for k in range(evecs.shape[1]):
        rho = state_density(evecs[:, k])
        vals[k] = float(rho[bulk_mask].sum())
    return vals


##--------------##
##Bondcurrents##
##--------------##
def bond_current_for_state(H, psi, i, j):
    return float(-2.0 * np.imag(np.conjugate(psi[i]) * H[i, j] * psi[j]))


def compute_currents_for_bonds(H, psi, bonds):
    vals = []
    for i, j in bonds:
        vals.append((i, j, bond_current_for_state(H, psi, i, j)))
    return vals


##----------------------##
##Interactiveviewerclass##
##----------------------##
class HaldaneEigenstateViewer:
    def __init__(
        self,
        sites,
        H,
        evals,
        evecs,
        nn_pairs,
        nnn_pairs_nu,
        edge_mask,
        bulk_mask,
        edge_parts,
        bulk_parts,
        edge_width,
        near_gap_count=80,
    ):
        self.sites = sites
        self.H = H
        self.evals = evals
        self.evecs = evecs
        self.nn_pairs = nn_pairs
        self.nnn_pairs_nu = nnn_pairs_nu
        self.nnn_pairs_plain = [(i, j) for i, j, _ in nnn_pairs_nu]
        self.edge_mask = edge_mask
        self.bulk_mask = bulk_mask
        self.edge_parts = edge_parts
        self.bulk_parts = bulk_parts
        self.edge_width = edge_width
        self.xy, self.sub = get_positions_array(sites)
        self.r = radial_distances(sites)
        self.rmax = self.r.max()

        self.order_absE = np.argsort(np.abs(self.evals))
        self.near_gap_count = min(near_gap_count, len(self.evals))
        self.near_gap_inds = self.order_absE[:self.near_gap_count]

        self.current_index = int(self.near_gap_inds[np.argmax(self.edge_parts[self.near_gap_inds])])

        self.fig = plt.figure(figsize=(14, 10))
        gs = self.fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.2], height_ratios=[1.0, 1.0])

        self.ax_spec = self.fig.add_subplot(gs[0, 0])
        self.ax_density = self.fig.add_subplot(gs[0, 1])
        self.ax_radial = self.fig.add_subplot(gs[0, 2])
        self.ax_nn = self.fig.add_subplot(gs[1, 0])
        self.ax_nnn = self.fig.add_subplot(gs[1, 1])
        self.ax_text = self.fig.add_subplot(gs[1, 2])

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.density_cbar = None
        
        self.draw_static_spectrum()
        self.update_all()

    def draw_static_spectrum(self):
        self.ax_spec.clear()
        x = np.arange(self.near_gap_count)
        y = self.evals[self.near_gap_inds]
        c = self.edge_parts[self.near_gap_inds]

        self.spec_scatter = self.ax_spec.scatter(x, y, c=c, s=60)
        self.ax_spec.axhline(0.0, linestyle="--", linewidth=1.0)
        self.ax_spec.set_xlabel("State rank by |E|")
        self.ax_spec.set_ylabel("Energy")
        self.ax_spec.set_title("Near-gap spectrum colored by edge participation")
        self.ax_spec.grid(True, linestyle="--", linewidth=0.5)

        self.spec_marker, = self.ax_spec.plot([], [], marker="o", markersize=12, markerfacecolor="none", markeredgewidth=2)

    def draw_density(self, k):
        self.ax_density.clear()
        psi = self.evecs[:, k]
        rho = state_density(psi)

        sc = self.ax_density.scatter(
            self.xy[:, 0],
            self.xy[:, 1],
            c=rho,
            s=35 + 300 * rho,
            edgecolors="none",
        )

        outer = plt.Circle((0.0, 0.0), self.rmax, fill=False, linestyle="--")
        inner = plt.Circle((0.0, 0.0), max(self.rmax - self.edge_width, 0.0), fill=False, linestyle=":")
        self.ax_density.add_artist(outer)
        self.ax_density.add_artist(inner)

        self.ax_density.set_aspect("equal")
        self.ax_density.set_xlabel("x")
        self.ax_density.set_ylabel("y")
        self.ax_density.set_title("State density")
        self.ax_density.grid(True, linestyle="--", linewidth=0.5)

        ##Createcolorbaronlyonce##
        if self.density_cbar is None:
            self.density_cbar = self.fig.colorbar(sc, ax=self.ax_density, fraction=0.046, pad=0.04)
            self.density_cbar.set_label(r"$|\psi_i|^2$")

    def draw_radial(self, k):
        self.ax_radial.clear()
        psi = self.evecs[:, k]
        rho = state_density(psi)
        order = np.argsort(self.r)

        self.ax_radial.plot(self.r[order], rho[order], marker="o", linestyle="None", markersize=4)
        self.ax_radial.set_xlabel("Radial distance from center")
        self.ax_radial.set_ylabel(r"$|\psi_i|^2$")
        self.ax_radial.set_title("Radial profile")
        self.ax_radial.grid(True, linestyle="--", linewidth=0.5)

    def draw_currents(self, ax, bonds, k, title):
        ax.clear()
        psi = self.evecs[:, k]
        currents = compute_currents_for_bonds(self.H, psi, bonds)
        absJ = np.array([abs(J) for _, _, J in currents], dtype=float)

        ax.scatter(self.xy[:, 0], self.xy[:, 1], s=10, alpha=0.25)

        if absJ.size > 0 and absJ.max() > 0:
            cutoff = 0.08 * absJ.max()
            for i, j, J in currents:
                if abs(J) < cutoff:
                    continue
                x0, y0 = self.xy[i]
                x1, y1 = self.xy[j]
                color = "red" if J > 0 else "blue"
                lw = 0.5 + 3.5 * abs(J) / absJ.max()
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=0.9)

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.5)

    def draw_text_panel(self, k):
        self.ax_text.clear()
        self.ax_text.axis("off")

        psi = self.evecs[:, k]
        rho = state_density(psi)

        lines = [
            "Phase 1 Haldane Dot Viewer",
            "",
            f"state index: {k}",
            f"energy: {self.evals[k]:+.6f}",
            f"|E| rank: {int(np.where(self.order_absE == k)[0][0])}",
            f"edge participation: {self.edge_parts[k]:.4f}",
            f"bulk participation: {self.bulk_parts[k]:.4f}",
            f"edge shell weight: {rho[self.edge_mask].sum():.4f}",
            f"bulk weight: {rho[self.bulk_mask].sum():.4f}",
            "",
            "Controls:",
            "Left / [  : previous state",
            "Right / ] : next state",
            "Up / Down : jump by 5",
            "Click point in spectrum to select",
        ]

        self.ax_text.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11)

    def update_spectrum_marker(self, k):
        idx = np.where(self.near_gap_inds == k)[0]
        if idx.size > 0:
            x = idx[0]
            y = self.evals[k]
            self.spec_marker.set_data([x], [y])
        else:
            self.spec_marker.set_data([], [])

    def update_all(self):
        k = self.current_index
        self.update_spectrum_marker(k)
        self.draw_density(k)
        self.draw_radial(k)
        self.draw_currents(self.ax_nn, self.nn_pairs, k, "NN bond currents")
        self.draw_currents(self.ax_nnn, self.nnn_pairs_plain, k, "NNN bond currents")
        self.draw_text_panel(k)

        self.fig.suptitle(
            f"Selected state = {k}, E = {self.evals[k]:+.6f}, edge participation = {self.edge_parts[k]:.4f}",
            fontsize=16
        )
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ["right", "]"]:
            self.current_index = min(self.current_index + 1, len(self.evals) - 1)
            self.update_all()
        elif event.key in ["left", "["]:
            self.current_index = max(self.current_index - 1, 0)
            self.update_all()
        elif event.key == "up":
            self.current_index = min(self.current_index + 5, len(self.evals) - 1)
            self.update_all()
        elif event.key == "down":
            self.current_index = max(self.current_index - 5, 0)
            self.update_all()

    def on_click(self, event):
        if event.inaxes != self.ax_spec:
            return
        if event.xdata is None or event.ydata is None:
            return

        x_click = int(np.round(event.xdata))
        x_click = max(0, min(x_click, self.near_gap_count - 1))
        self.current_index = int(self.near_gap_inds[x_click])
        self.update_all()


##------##
##Main##
##------##
if __name__ == "__main__":

    ##Phase1parameters##
    R = 8.0
    edge_width = 1.25
    t = 1.0
    t2 = 0.1
    phi = np.pi / 2.0
    M = 0.2

    ##Buildgeometry##
    sites = build_disk_honeycomb_dot(R=R, pad=4)
    nn_pairs = find_nn_pairs(sites)
    nnn_pairs_nu = find_nnn_pairs_with_nu(sites)

    ##BuildHamiltonian##
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
    print(f"N sites = {len(sites)}")
    print(f"N NN bonds = {len(nn_pairs)}")
    print(f"N NNN bonds = {len(nnn_pairs_nu)}")
    print(f"Hermiticity max error = {herm_err:.3e}")

    ##Solve##
    evals, evecs = np.linalg.eigh(H)

    ##Metrics##
    edge_mask = make_edge_mask(sites, edge_width=edge_width)
    bulk_mask = make_bulk_mask(sites, edge_width=edge_width)
    edge_parts = edge_participation_all(evecs, edge_mask)
    bulk_parts = bulk_participation_all(evecs, bulk_mask)

    ##Launchviewer##
    viewer = HaldaneEigenstateViewer(
        sites=sites,
        H=H,
        evals=evals,
        evecs=evecs,
        nn_pairs=nn_pairs,
        nnn_pairs_nu=nnn_pairs_nu,
        edge_mask=edge_mask,
        bulk_mask=bulk_mask,
        edge_parts=edge_parts,
        bulk_parts=bulk_parts,
        edge_width=edge_width,
        near_gap_count=80,
    )

    plt.show()