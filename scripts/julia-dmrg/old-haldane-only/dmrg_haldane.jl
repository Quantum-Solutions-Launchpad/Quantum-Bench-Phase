using ITensors
using ITensorMPS
using Random
using JSON

function print_usage()
    println(
        """
        Usage:
          julia scripts/julia-dmrg/old-haldane-only/dmrg_haldane.jl [options]

        Options:
          --n-sites 4,6,8              Comma-separated site counts
          --t2 0.0,0.5,1.0             Comma-separated t2 values
          --maxdims 20,50,100,200|50,100,200,400
                                       One or more maxdim schedules separated by |
          --n-occ all                  Use all particle-number sectors
          --n-occ 0,2,4,6              Or a comma-separated sector subset
          --seeds 1234,1235            Repeat runs with different initial-state seeds
          --nsweeps 4
          --cutoff 1e-9
          --t1 1.0
          --phi 0.7853981633974483
          --conserve-qns true
          --output cache/.../file.json
        """
    )
end

function parse_int_list(text::AbstractString)
    isempty(strip(text)) && error("Expected a comma-separated integer list, got an empty string.")
    return [parse(Int, strip(part)) for part in split(text, ",") if !isempty(strip(part))]
end

function parse_float_list(text::AbstractString)
    isempty(strip(text)) && error("Expected a comma-separated float list, got an empty string.")
    return [parse(Float64, strip(part)) for part in split(text, ",") if !isempty(strip(part))]
end

function parse_bool(text::AbstractString)
    value = lowercase(strip(text))
    if value in ("true", "1", "yes")
        return true
    elseif value in ("false", "0", "no")
        return false
    end
    error("Could not parse boolean value: $text")
end

function parse_maxdim_schedules(text::AbstractString)
    schedules = Vector{Vector{Int}}()
    for chunk in split(text, "|")
        schedule = parse_int_list(chunk)
        isempty(schedule) && error("Each maxdim schedule must contain at least one integer.")
        push!(schedules, schedule)
    end
    isempty(schedules) && error("Expected at least one maxdim schedule.")
    return schedules
end

function parse_args(args::Vector{String})
    config = Dict{String, Any}(
        "n_sites_values" => [6],
        "t1" => 1.0,
        "t2_values" => [2.0],
        "phi" => pi / 4,
        "spin" => 2,
        "n_occ_spec" => "all",
        "nsweeps" => 4,
        "maxdim_schedules" => [[20, 50, 100, 200]],
        "cutoff" => 1e-9,
        "conserve_qns" => true,
        "seeds" => [1234],
        "output" => nothing,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            print_usage()
            exit(0)
        elseif arg == "--n-sites"
            i += 1
            config["n_sites_values"] = parse_int_list(args[i])
        elseif arg == "--t1"
            i += 1
            config["t1"] = parse(Float64, args[i])
        elseif arg == "--t2"
            i += 1
            config["t2_values"] = parse_float_list(args[i])
        elseif arg == "--phi"
            i += 1
            config["phi"] = parse(Float64, args[i])
        elseif arg == "--n-occ"
            i += 1
            config["n_occ_spec"] = args[i]
        elseif arg == "--nsweeps"
            i += 1
            config["nsweeps"] = parse(Int, args[i])
        elseif arg == "--maxdims"
            i += 1
            config["maxdim_schedules"] = parse_maxdim_schedules(args[i])
        elseif arg == "--cutoff"
            i += 1
            config["cutoff"] = parse(Float64, args[i])
        elseif arg == "--conserve-qns"
            i += 1
            config["conserve_qns"] = parse_bool(args[i])
        elseif arg == "--seeds"
            i += 1
            config["seeds"] = parse_int_list(args[i])
        elseif arg == "--output"
            i += 1
            config["output"] = args[i]
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end

    return config
end

function resolve_n_occ_values(spec::AbstractString, n_sites::Int, spin::Int)
    max_n_occ = spin * n_sites
    if lowercase(strip(spec)) == "all"
        return collect(0:max_n_occ)
    end

    n_occ_values = parse_int_list(spec)
    for n_occ in n_occ_values
        0 <= n_occ <= max_n_occ || error("n_occ=$n_occ is outside 0:$max_n_occ")
    end
    return sort(unique(n_occ_values))
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

    return (
        sites=sites,
        H=H,
        profile=profile,
    )
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

function default_output_path(repo_root::String)
    outdir = joinpath(repo_root, "cache", "haldane-model", "real-space", "dmrg")
    mkpath(outdir)
    return joinpath(outdir, "dmrg-sweep.json")
end

function main()
    config = parse_args(ARGS)
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    outfile = something(config["output"], default_output_path(repo_root))
    outdir = dirname(outfile)
    mkpath(outdir)

    runs = Dict{String, Any}[]
    overall_t0 = time_ns()

    for n_sites in config["n_sites_values"]
        n_occ_values = resolve_n_occ_values(config["n_occ_spec"], n_sites, config["spin"])
        for t2 in config["t2_values"]
            for maxdim_schedule in config["maxdim_schedules"]
                for seed in config["seeds"]
                    println(
                        "Starting run: n_sites=$n_sites, t2=$t2, maxdim=$(join(maxdim_schedule, ",")), seed=$seed"
                    )

                    run_t0 = time_ns()
                    hamiltonian = build_ring_hamiltonian_spinful(
                        n_sites;
                        t1=config["t1"],
                        t2=t2,
                        phi=config["phi"],
                        conserve_qns=config["conserve_qns"],
                    )
                    sites = hamiltonian.sites
                    H = hamiltonian.H

                    sectors = Dict{String, Any}[]
                    energies = Float64[]
                    dmrg_time_total = 0.0

                    for n_occ in n_occ_values
                        result = run_dmrg_ring_spinful_check(
                            sites,
                            H;
                            n_occ=n_occ,
                            nsweeps=config["nsweeps"],
                            maxdim=maxdim_schedule,
                            cutoff=config["cutoff"],
                            seed=seed + n_occ,
                        )

                        push!(energies, result.energy)
                        push!(
                            sectors,
                            Dict(
                                "n_occ" => n_occ,
                                "seed" => seed + n_occ,
                                "energy" => result.energy,
                                "profile" => result.profile,
                                "link_dims" => result.link_dims,
                                "max_link_dim" => result.max_link_dim,
                                "avg_link_dim" => result.avg_link_dim,
                            ),
                        )
                        dmrg_time_total += result.profile["dmrg"]["elapsed_s"]
                        println(
                            "  n_occ=$n_occ  E=$(result.energy)  dmrg_s=$(result.profile["dmrg"]["elapsed_s"])  max_link_dim=$(result.max_link_dim)"
                        )
                    end

                    total_wall_time_s = (time_ns() - run_t0) / 1e9
                    push!(
                        runs,
                        Dict(
                            "n_sites" => n_sites,
                            "t1" => config["t1"],
                            "t2" => t2,
                            "phi" => config["phi"],
                            "conserve_qns" => config["conserve_qns"],
                            "nsweeps" => config["nsweeps"],
                            "cutoff" => config["cutoff"],
                            "seed" => seed,
                            "n_occ_values" => n_occ_values,
                            "maxdim_schedule" => maxdim_schedule,
                            "hamiltonian_profile" => hamiltonian.profile,
                            "energies" => energies,
                            "sectors" => sectors,
                            "summary" => Dict(
                                "num_sectors" => length(sectors),
                                "total_wall_time_s" => total_wall_time_s,
                                "dmrg_wall_time_s" => dmrg_time_total,
                                "hamiltonian_build_wall_time_s" => sum(
                                    section["elapsed_s"] for section in values(hamiltonian.profile)
                                ),
                                "max_link_dim_over_sectors" => isempty(sectors) ? 0 : maximum(
                                    sector["max_link_dim"] for sector in sectors
                                ),
                            ),
                        ),
                    )

                    println("Completed run in $(total_wall_time_s) s")
                end
            end
        end
    end

    out = Dict(
        "format" => "dmrg_haldane_sweep_v1",
        "created_by" => "scripts/julia-dmrg/old-haldane-only/dmrg_haldane.jl",
        "parameters" => Dict(
            "n_sites_values" => config["n_sites_values"],
            "t1" => config["t1"],
            "t2_values" => config["t2_values"],
            "phi" => config["phi"],
            "spin" => config["spin"],
            "n_occ_spec" => config["n_occ_spec"],
            "nsweeps" => config["nsweeps"],
            "maxdim_schedules" => config["maxdim_schedules"],
            "cutoff" => config["cutoff"],
            "conserve_qns" => config["conserve_qns"],
            "seeds" => config["seeds"],
        ),
        "runs" => runs,
        "summary" => Dict(
            "num_runs" => length(runs),
            "total_wall_time_s" => (time_ns() - overall_t0) / 1e9,
        ),
    )

    open(outfile, "w") do io
        JSON.print(io, out, 2)
    end

    println("Wrote $outfile")
end

main()
