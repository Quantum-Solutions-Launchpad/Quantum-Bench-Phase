import quaph

result = quaph.run_simulated_ideal(
    "haldane-hubbard",
    n_sites=6,
    vqe_iters=10000,
    vqe_layers=5,
    vqe_reps=10,
    iqpe_time=0.2,
    iqpe_trot=5,
    iqpe_iters=8,
    iqpe_reps=20,
    log_dir="logs",
    plot_dir="plots",
)
