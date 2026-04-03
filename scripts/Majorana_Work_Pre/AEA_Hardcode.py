import numpy as np

class SimpleBdGHamiltonian:
    def __init__(self, M, Delta, constant=0.0):
        ##validate input##
        assert M.shape == Delta.shape, "M and Delta must have the same shape"
        assert np.allclose(M, M.conj().T), "M must be Hermitian"
        assert np.allclose(Delta, -Delta.T), "Delta must be antisymmetric"

        self.M = M
        self.Delta = Delta
        self.constant = constant
        self.n = M.shape[0]

    def build_bdg_matrix(self):
        M, Delta, n = self.M, self.Delta, self.n
        top = np.hstack((M, Delta))
        bottom = np.hstack((-Delta.conj(), -M.T))
        return np.vstack((top, bottom))

    def diagonalize(self):
        H_bdg = self.build_bdg_matrix()
        evals, evecs = np.linalg.eigh(H_bdg)
        return np.sort(np.real(evals)), evecs


##Kitaev chain function##
def generate_kitaev_chain(n, t, delta, mu):
    M = np.zeros((n, n), dtype=complex)
    Delta = np.zeros((n, n), dtype=complex)

    for i in range(n):
        M[i, i] = -mu
        if i < n - 1:
            M[i, i+1] = M[i+1, i] = -t
            Delta[i, i+1] = delta
            Delta[i+1, i] = -delta

    return SimpleBdGHamiltonian(M, Delta)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n = 6
    t = 1.0
    delta = 1.0
    mus = np.linspace(0.0, 3.0, 100)
    eps0, eps1 = [], []

    for mu in mus:
        H = generate_kitaev_chain(n, t, delta, mu)
        evals, _ = H.diagonalize()
        near_zero = evals[np.argsort(np.abs(evals))][:4]
        near_zero.sort()
        eps0.append(near_zero[0])
        eps1.append(near_zero[1])

    eps0 = np.array(eps0)
    eps1 = np.array(eps1)

    plt.figure(figsize=(7, 5))
    plt.plot(mus, eps0, label=r"$\epsilon_0$", color='black')
    plt.plot(mus, -eps0, color='black')
    plt.plot(mus, eps1, label=r"$\epsilon_1$", linestyle='--', color='gray')
    plt.plot(mus, -eps1, linestyle='--', color='gray')
    plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
    plt.xlabel("Chemical Potential (μ)")
    plt.ylabel("Excitation Energy")
    plt.title("Ideal BdG Spectrum – Central 4 Modes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
