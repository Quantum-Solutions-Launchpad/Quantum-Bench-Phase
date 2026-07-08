# Precompile workload for the DMRG sysimage: run the real CLI end-to-end on
# tiny spinless (Haldane-like) and spinful (Hubbard-like) Hamiltonians so all
# hot code paths (OpSum/MPO with QNs, complex coefficients, dmrg, observable
# expectation, JSON I/O) get compiled into the image.

const HERE = @__DIR__
const CLI = joinpath(HERE, "..", "dmrg_itensor_cli.jl")

function run_cli(args::Vector{String})
    empty!(ARGS)
    append!(ARGS, args)
    # include() re-runs main(); method redefinition warnings are harmless here.
    Base.include(Main, CLI)
end

tmp = mktempdir()

# Spinless fermions, complex hoppings, QNs on and off, both initial states
run_cli([
    "--hamiltonian", joinpath(HERE, "sample_hamiltonian_spinless.json"),
    "--output", joinpath(tmp, "spinless_qns.json"),
    "--nsweeps", "2", "--maxdims", "10,20", "--seed", "1",
    "--conserve-qns", "true", "--conserve-sz", "true",
    "--initial-state", "packed",
])
run_cli([
    "--hamiltonian", joinpath(HERE, "sample_hamiltonian_spinless.json"),
    "--output", joinpath(tmp, "spinless_noqns.json"),
    "--nsweeps", "2", "--maxdims", "10,20", "--seed", "1",
    "--conserve-qns", "false", "--conserve-sz", "false",
    "--initial-state", "packed",
])

# Spinful electrons with an observable MPO, both Sz settings and both seeds
run_cli([
    "--hamiltonian", joinpath(HERE, "sample_hamiltonian_spinful.json"),
    "--output", joinpath(tmp, "spinful_sz.json"),
    "--nsweeps", "2", "--maxdims", "10,20", "--seed", "1",
    "--conserve-qns", "true", "--conserve-sz", "true",
    "--initial-state", "neel",
])
run_cli([
    "--hamiltonian", joinpath(HERE, "sample_hamiltonian_spinful.json"),
    "--output", joinpath(tmp, "spinful_nosz.json"),
    "--nsweeps", "2", "--maxdims", "10,20", "--seed", "1",
    "--conserve-qns", "true", "--conserve-sz", "false",
    "--initial-state", "packed",
])

println("precompile workload complete")
