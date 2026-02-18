using ITensors
using ITensorMPS
using Random #for shuffle! to randomize inital product state
using JSON

#=Runs DMRG on a tight-binding ring (NN+complex NN hoppings) 

=#
function run_dmrg_ring_spinful_check(n_sites::Int;
    t1::Float64=1.0, #nearest-neighbor amplitude
    t2::Float64=0.05, #next nearest neighbor hopping amplitude
    phi::Float64=pi/4, #phase angle
    n_occ::Int=4, #total number of electrons in system
    nsweeps::Int=6,
    maxdim::Vector{Int}=[20, 50, 100, 200, 200, 200],
    cutoff::Float64=1e-14,
    conserve_qns::Bool=true #tensors become block-sparse, speedup
)
    #construct list of site indices in local Hilbert space
    sites = siteinds("Electron", n_sites; conserve_qns=conserve_qns)
    os = OpSum()

    for i in 1:n_sites
        #loop over each real-space site on the ring
        j1 = (i % n_sites) + 1 #nearest neighbor i+1
        j2 = ((i + 1) % n_sites) + 1 #next nearest neighbor i+2

        # NN hopping for both spins (up, down).
        # - "Cup" annihilates an up electron
        # - "Cdagup" creates an up electron 
        # Same logic applies for spin down
        for (cdag, c) in (("Cdagup","Cup"), ("Cdagdn","Cdn"))
            os += (-t1, cdag, i,  c, j1)
            os += (-t1, cdag, j1, c, i)
            # Hermitian conjugate: -t1 cdag_{j1,σ} c_{i,σ}
        end
        # NNN complex phase of hopping for both spins
        phase = cis(phi)
        for (cdag, c) in (("Cdagup","Cup"), ("Cdagdn","Cdn"))
            os += (-t2 * phase, cdag, i,  c, j2)
            # -t2 e^{i psi} cdag_{i,σ} c_{j2,σ}
            os += (-t2 * conj(phase), cdag, j2, c, i)
            # -t2 e^{-i psi} cdag_{j2,σ} c_{i,σ}
        end
    end

    # convert to MPO, the tensor-network representation of the Hamiltonian
    H = MPO(os, sites)

    # Build product state with exactly n_occ electrons total
    state = fill("Emp", n_sites) #initalize all empty sites
    remaining = n_occ
    for i in 1:n_sites
        #fill sites one by one until we place exactly n_occ electrons
        if remaining >= 2
            state[i] = "UpDn"
            remaining -= 2
        elseif remaining == 1
            state[i] = "Up"
            remaining -= 1
        end
    end
    shuffle!(state) #randomly permute site occupation

    psi0 = productMPS(sites, state)
    #DMRG algo finds approx ground state of H
    energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
    return energy
end


n_sites = 4
t1, t2, phi = 1.0, 0.05, pi/4
spin = 2
max_n_occ = spin * n_sites

energies = Float64[]
for n_occ in 0:max_n_occ #loop over all possible particle numbers 0-8. For each particle number, compute GSE
    E = run_dmrg_ring_spinful_check(n_sites; t1=t1, t2=t2, phi=phi, n_occ=n_occ) #call DMRG with this particle number n_occ
    push!(energies, E)
    #append computed energy to energies array
    println("n_occ=$n_occ  E=$E")
end

out = Dict(
    "n_sites" => n_sites,
    "t1" => t1,
    "t2" => t2,
    "phi" => phi,
    "energies" => energies
)

repo_root = normpath(joinpath(@__DIR__, "..", ".."))
outdir = joinpath(repo_root, "cache", "haldane-model", "real-space", "dmrg")
mkpath(outdir)
outfile = joinpath(outdir, "dmrg-ideal-t2-$(t2).json")

open(outfile, "w") do io
    JSON.print(io, out)
end

println("Wrote $outfile")
