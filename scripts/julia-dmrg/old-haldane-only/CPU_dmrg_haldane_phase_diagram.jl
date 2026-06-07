#!/usr/bin/env julia
#=
CPU_dmrg_haldane_phase_diagram.jl
Created by Maggie Bao 3/23/2026

Run a CPU DMRG Haldane-model phase-diagram sweep over:
  n_sites in [N_start, N_end] with step n_step
  t2 in [t2_start, t2_end] with step t2_step

The parameter points are sharded across SLURM ranks in round-robin order.
Results are saved incrementally as JSONL batches.

Usage:
  julia --project=/pscratch/sd/m/mbao202/NNL-P7/scripts/julia-dmrg \
    /pscratch/sd/m/mbao202/NNL-P7/scripts/julia-dmrg/old-haldane-only/CPU_dmrg_haldane_phase_diagram.jl \
    N_start N_end t1 t2_start t2_end t2_step [label]
=#

using Dates
using JSON
using Printf
using Random
using ITensors
using ITensorMPS

if length(ARGS) < 6
    println(
        "Usage: julia CPU_dmrg_haldane_phase_diagram.jl N_start N_end t1 t2_start t2_end t2_step [label]"
    )
    exit(1)
end

N_start  = parse(Int, ARGS[1])
N_end    = parse(Int, ARGS[2])
t1       = parse(Float64, ARGS[3])
t2_start = parse(Float64, ARGS[4])
t2_end   = parse(Float64, ARGS[5])
t2_step  = parse(Float64, ARGS[6])
label    = length(ARGS) >= 7 ? ARGS[7] : "run"
n_step   = parse(Int, get(ENV, "DMRG_N_STEP", "1"))
n_step > 0 || error("DMRG_N_STEP must be positive.")

phi = parse(Float64, get(ENV, "DMRG_PHI", string(pi / 4)))
nsweeps = parse(Int, get(ENV, "DMRG_NSWEEPS", "4"))
maxdim = begin
    md = get(ENV, "DMRG_MAXDIM", "20,50,100,200")
    parse.(Int, split(md, ','))
end
cutoff = parse(Float64, get(ENV, "DMRG_CUTOFF", "1e-9"))
conserve_qns = lowercase(get(ENV, "DMRG_CONSERVE_QNS", "true")) in ("true", "1", "yes")
n_occ_spec = get(ENV, "DMRG_N_OCC", "all")
seed_values = begin
    raw = get(ENV, "DMRG_SEEDS", "1234")
    parse.(Int, split(raw, ','))
end
SAVE_BATCH_SIZE = parse(Int, get(ENV, "DMRG_SAVE_BATCH", "10"))

const RANK = Ref(0)
const SIZE = Ref(1)

if haskey(ENV, "SLURM_PROCID")
    RANK[] = parse(Int, ENV["SLURM_PROCID"])
    if haskey(ENV, "SLURM_NTASKS")
        SIZE[] = parse(Int, ENV["SLURM_NTASKS"])
    elseif haskey(ENV, "SLURM_NTASKS_PER_NODE") && haskey(ENV, "SLURM_JOB_NUM_NODES")
        tasks_per_node = parse(Int, ENV["SLURM_NTASKS_PER_NODE"])
        num_nodes = parse(Int, ENV["SLURM_JOB_NUM_NODES"])
        SIZE[] = tasks_per_node * num_nodes
    else
        SIZE[] = 1
    end
else
    RANK[] = 0
    SIZE[] = 1
    @warn "Not running under SLURM - executing in single process mode"
end

function resolve_n_occ_values(spec::AbstractString, n_sites::Int)
    max_n_occ = 2 * n_sites
    if lowercase(strip(spec)) == "all"
        return collect(0:max_n_occ)
    end

    values = sort(unique(parse.(Int, split(spec, ","))))
    for n_occ in values
        0 <= n_occ <= max_n_occ || error("n_occ=$n_occ is outside 0:$max_n_occ")
    end
    return values
end

function profile_section!(f::Function, profile::Dict{String, Dict{String, Float64}}, label::String)
    t0 = time_ns()
    result = nothing
    alloc = @allocated begin
        result = f()
    end
    elapsed_s = (time_ns() - t0) / 1e9
    profile[label] = Dict(
        "elapsed_s" => elapsed_s,
        "alloc_bytes" => Float64(alloc),
    )
    return result
end

function build_ring_hamiltonian_spinful(n_sites::Int;
    t1::Float64=1.0,
    t2::Float64=0.05,
    phi::Float64=pi / 4,
    conserve_qns::Bool=true
)
    profile = Dict{String, Dict{String, Float64}}()

    sites = profile_section!(profile, "siteinds") do
        siteinds("Electron", n_sites; conserve_qns=conserve_qns)
    end
    os = OpSum()

    profile_section!(profile, "build_opsum") do
        phase = cis(phi)
        phase_conj = conj(phase)
        for i in 1:n_sites
            j1 = (i % n_sites) + 1
            j2 = ((i + 1) % n_sites) + 1

            os += (-t1, "Cdagup", i, "Cup", j1)
            os += (-t1, "Cdagup", j1, "Cup", i)
            os += (-t1, "Cdagdn", i, "Cdn", j1)
            os += (-t1, "Cdagdn", j1, "Cdn", i)

            os += (-t2 * phase, "Cdagup", i, "Cup", j2)
            os += (-t2 * phase_conj, "Cdagup", j2, "Cup", i)
            os += (-t2 * phase, "Cdagdn", i, "Cdn", j2)
            os += (-t2 * phase_conj, "Cdagdn", j2, "Cdn", i)
        end
    end

    H = profile_section!(profile, "build_mpo") do
        MPO(os, sites)
    end

    return (sites=sites, H=H, profile=profile)
end

function collect_link_dims(psi)
    dims = Int[]
    for bond in 1:(length(psi) - 1)
        link = linkind(psi, bond)
        if !isnothing(link)
            push!(dims, dim(link))
        end
    end
    return dims
end

function run_dmrg_ring_spinful_check(sites, H;
    n_occ::Int=length(sites),
    nsweeps::Int=4,
    maxdim::Vector{Int}=[20, 50, 100, 200],
    cutoff::Float64=1e-9,
    seed::Int=1234
)
    n_sites = length(sites)
    profile = Dict{String, Dict{String, Float64}}()
    rng = MersenneTwister(seed)

    state = profile_section!(profile, "build_init_state") do
        local_state = fill("Emp", n_sites)
        remaining = n_occ
        for i in 1:n_sites
            if remaining >= 2
                local_state[i] = "UpDn"
                remaining -= 2
            elseif remaining == 1
                local_state[i] = "Up"
                remaining -= 1
            end
        end
        shuffle!(rng, local_state)
        local_state
    end

    psi0 = profile_section!(profile, "build_product_mps") do
        productMPS(sites, state)
    end

    dmrg_result = profile_section!(profile, "dmrg") do
        local_energy, local_psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
        (energy=local_energy, psi=local_psi)
    end

    link_dims = collect_link_dims(dmrg_result.psi)
    max_link_dim = isempty(link_dims) ? 0 : maximum(link_dims)
    avg_link_dim = isempty(link_dims) ? 0.0 : sum(link_dims) / length(link_dims)

    return (
        energy=dmrg_result.energy,
        profile=profile,
        link_dims=link_dims,
        max_link_dim=max_link_dim,
        avg_link_dim=avg_link_dim,
    )
end

function build_t2_values(start_value::Float64, end_value::Float64, step_value::Float64)
    step_value > 0 || error("t2_step must be positive.")
    values = Float64[]
    current = start_value
    atol = max(1e-12, abs(step_value) * 1e-9)
    while current <= end_value + atol
        push!(values, round(current; digits=12))
        current += step_value
    end
    return values
end

all_points = NamedTuple{(:n_sites, :t2)}[]
for n_sites in N_start:n_step:N_end
    for t2 in build_t2_values(t2_start, t2_end, t2_step)
        push!(all_points, (n_sites=n_sites, t2=t2))
    end
end

local_points = NamedTuple{(:n_sites, :t2)}[]
for (i, point) in enumerate(all_points)
    if ((i - 1) % SIZE[]) == RANK[]
        push!(local_points, point)
    end
end

@info "Process identification" rank=RANK[] size=SIZE[] pid=getpid() host=gethostname()
@info "Ensemble config" N_start N_end n_step t1 t2_start t2_end t2_step label nsweeps maxdim cutoff phi conserve_qns n_occ_spec seeds=seed_values save_batch=SAVE_BATCH_SIZE
@info "Rank $(RANK[]) assigned $(length(local_points)) parameter points" first_few=local_points[1:min(end, 5)]

out_dir = joinpath("cache", "haldane-model", "real-space", "dmrg", label)
isdir(out_dir) || mkpath(out_dir)
timestamp = Dates.format(now(), "yyyy-mm-dd-HHMMSS")
outfile = joinpath(out_dir, "rank$(RANK[])_of_$(SIZE[])_$(timestamp).jsonl")

@info "Rank $(RANK[]) output file: $outfile"

results_buffer = Vector{Dict{String, Any}}()

for (idx, point) in enumerate(local_points)
    n_sites = point.n_sites
    t2 = point.t2

    try
        @info "Rank $(RANK[]): Starting n_sites=$n_sites, t2=$t2 ($(idx)/$(length(local_points)))"

        run_t0 = time_ns()
        hamiltonian = build_ring_hamiltonian_spinful(
            n_sites;
            t1=t1,
            t2=t2,
            phi=phi,
            conserve_qns=conserve_qns,
        )
        sites = hamiltonian.sites
        H = hamiltonian.H
        n_occ_values = resolve_n_occ_values(n_occ_spec, n_sites)

        sectors = Dict{String, Any}[]
        energies = Float64[]
        dmrg_time_total = 0.0

        for seed in seed_values
            for n_occ in n_occ_values
                result = run_dmrg_ring_spinful_check(
                    sites,
                    H;
                    n_occ=n_occ,
                    nsweeps=nsweeps,
                    maxdim=maxdim,
                    cutoff=cutoff,
                    seed=seed + n_occ,
                )

                push!(energies, result.energy)
                push!(
                    sectors,
                    Dict(
                        "n_occ" => n_occ,
                        "seed" => seed + n_occ,
                        "base_seed" => seed,
                        "energy" => @sprintf("%.17g", result.energy),
                        "profile" => result.profile,
                        "link_dims" => result.link_dims,
                        "max_link_dim" => result.max_link_dim,
                        "avg_link_dim" => result.avg_link_dim,
                    ),
                )
                dmrg_time_total += result.profile["dmrg"]["elapsed_s"]
            end
        end

        total_wall_time_s = (time_ns() - run_t0) / 1e9
        rec = Dict(
            "n_sites" => n_sites,
            "t1" => t1,
            "t2" => @sprintf("%.17g", t2),
            "phi" => phi,
            "conserve_qns" => conserve_qns,
            "nsweeps" => nsweeps,
            "maxdim" => maxdim,
            "cutoff" => cutoff,
            "n_occ_spec" => n_occ_spec,
            "n_occ_values" => n_occ_values,
            "seeds" => seed_values,
            "hamiltonian_profile" => hamiltonian.profile,
            "energies" => [@sprintf("%.17g", energy) for energy in energies],
            "sectors" => sectors,
            "summary" => Dict(
                "num_sectors" => length(sectors),
                "total_wall_time_s" => @sprintf("%.17g", total_wall_time_s),
                "dmrg_wall_time_s" => @sprintf("%.17g", dmrg_time_total),
                "hamiltonian_build_wall_time_s" => @sprintf(
                    "%.17g",
                    sum(section["elapsed_s"] for section in values(hamiltonian.profile)),
                ),
                "max_link_dim_over_sectors" => isempty(sectors) ? 0 : maximum(
                    sector["max_link_dim"] for sector in sectors
                ),
            ),
            "ts" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        )
        push!(results_buffer, rec)

        @info "Rank $(RANK[]): Completed n_sites=$n_sites, t2=$t2, wall=$(total_wall_time_s)s"
    catch e
        @error "Rank $(RANK[]): Error for n_sites=$n_sites, t2=$t2" exception=(e, catch_backtrace())

        err = Dict(
            "n_sites" => n_sites,
            "t1" => t1,
            "t2" => @sprintf("%.17g", t2),
            "phi" => phi,
            "error" => string(e),
            "ts" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        )
        push!(results_buffer, err)
    end

    if length(results_buffer) >= SAVE_BATCH_SIZE || idx == length(local_points)
        @info "Rank $(RANK[]): Saving $(length(results_buffer)) results to disk"
        open(outfile, "a") do io
            for record in results_buffer
                write(io, JSON.json(record))
                write(io, '\n')
            end
        end
        empty!(results_buffer)
        @info "Rank $(RANK[]): Save complete, continuing..."
    end
end

if !isempty(results_buffer)
    @info "Rank $(RANK[]): Final save of $(length(results_buffer)) results"
    open(outfile, "a") do io
        for record in results_buffer
            write(io, JSON.json(record))
            write(io, '\n')
        end
    end
end

@info "Rank $(RANK[]) finished all calculations. Output => $outfile"
