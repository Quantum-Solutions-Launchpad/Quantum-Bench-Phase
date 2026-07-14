using ITensors
using ITensorMPS
using JSON
using Random

function parse_bool(text::AbstractString)
    value = lowercase(strip(text))
    if value in ("true", "1", "yes")
        return true
    elseif value in ("false", "0", "no")
        return false
    end
    error("Could not parse boolean value: $text")
end

function parse_int_list(text::AbstractString)
    return [parse(Int, strip(part)) for part in split(text, ",") if !isempty(strip(part))]
end

function parse_args(args::Vector{String})
    config = Dict{String, Any}(
        "hamiltonian" => nothing,
        "output" => nothing,
        "nsweeps" => 4,
        "maxdims" => [20, 50, 100, 200],
        "cutoff" => 1e-9,
        "seed" => 1234,
        "conserve_qns" => true,
        "conserve_sz" => true,
        "initial_state" => "packed",
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--hamiltonian"
            i += 1
            config["hamiltonian"] = args[i]
        elseif arg == "--output"
            i += 1
            config["output"] = args[i]
        elseif arg == "--nsweeps"
            i += 1
            config["nsweeps"] = parse(Int, args[i])
        elseif arg == "--maxdims"
            i += 1
            config["maxdims"] = parse_int_list(args[i])
        elseif arg == "--cutoff"
            i += 1
            config["cutoff"] = parse(Float64, args[i])
        elseif arg == "--seed"
            i += 1
            config["seed"] = parse(Int, args[i])
        elseif arg == "--conserve-qns"
            i += 1
            config["conserve_qns"] = parse_bool(args[i])
        elseif arg == "--conserve-sz"
            i += 1
            config["conserve_sz"] = parse_bool(args[i])
        elseif arg == "--initial-state"
            i += 1
            config["initial_state"] = lowercase(strip(args[i]))
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end
    isnothing(config["hamiltonian"]) && error("--hamiltonian is required")
    isnothing(config["output"]) && error("--output is required")
    config["initial_state"] in ("packed", "neel") ||
        error("--initial-state must be packed or neel")
    return config
end

function complex_coeff(value)
    return complex(Float64(value["re"]), Float64(value["im"]))
end

function spinful_op(base::AbstractString, spin_orbital::Int)
    spin = spin_orbital % 2
    if base == "+"
        return spin == 0 ? "Cdagup" : "Cdagdn"
    elseif base == "-"
        return spin == 0 ? "Cup" : "Cdn"
    end
    error("Unsupported operator token: $base")
end

function spinless_op(base::AbstractString)
    if base == "+"
        return "Cdag"
    elseif base == "-"
        return "C"
    end
    error("Unsupported operator token: $base")
end

function add_term(os::OpSum, coeff, label::AbstractString, spin::Int)
    tokens = split(label)
    if length(tokens) == 0
        os += coeff
        return os
    end
    if length(tokens) % 2 != 0
        error("Malformed FermionicOp label: $label")
    end

    args = Any[coeff]
    for token in tokens
        pieces = split(token, "_")
        length(pieces) == 2 || error("Malformed FermionicOp token: $token")
        base = String(pieces[1])
        orb = parse(Int, pieces[2])
        site = div(orb, spin) + 1
        opname = spin == 2 ? spinful_op(base, orb) : spinless_op(base)
        push!(args, opname)
        push!(args, site)
    end
    os += tuple(args...)
    return os
end

function build_spinful_packed_state(n_sites::Int, n_occ::Int, rng)
    state = fill("Emp", n_sites)
    remaining = n_occ
    for i in 1:n_sites
        if remaining >= 2
            state[i] = "UpDn"
            remaining -= 2
        elseif remaining == 1
            state[i] = "Up"
            remaining -= 1
        end
    end
    shuffle!(rng, state)
    return state
end

function build_spinful_neel_state(n_sites::Int, n_occ::Int, rng)
    state = fill("Emp", n_sites)
    site_order = collect(1:n_sites)
    shuffle!(rng, site_order)

    # Place one electron on each selected site before creating any doublons.
    # Site parity follows the A/B ordering of the Hubbard lattice and seeds
    # staggered Up/Dn order at half filling.
    singly_occupied = min(n_occ, n_sites)
    for site in site_order[1:singly_occupied]
        state[site] = isodd(site) ? "Up" : "Dn"
    end

    extra = max(0, n_occ - n_sites)
    for site in site_order[1:extra]
        state[site] = "UpDn"
    end
    return state
end

function build_initial_state(
    n_sites::Int,
    spin::Int,
    n_occ::Int,
    seed::Int,
    strategy::AbstractString,
)
    0 <= n_occ <= spin * n_sites ||
        error("n_occ=$n_occ is outside the valid range 0:$(spin * n_sites)")
    rng = MersenneTwister(seed)
    if spin == 2
        if strategy == "neel"
            return build_spinful_neel_state(n_sites, n_occ, rng)
        elseif strategy == "packed"
            return build_spinful_packed_state(n_sites, n_occ, rng)
        end
        error("Unsupported spinful initial-state strategy: $strategy")
    elseif spin == 1
        state = fill("Emp", n_sites)
        for i in 1:min(n_occ, n_sites)
            state[i] = "Occ"
        end
        shuffle!(rng, state)
        return state
    end
    error("Unsupported spin=$spin")
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

function main()
    overall_t0 = time_ns()
    config = parse_args(ARGS)
    spec = JSON.parsefile(config["hamiltonian"])
    spin = Int(spec["spin"])
    n_sites = Int(spec["n_sites"])
    n_occ = Int(spec["n_occ"])

    sites = if spin == 2
        siteinds(
            "Electron",
            n_sites;
            conserve_nf=config["conserve_qns"],
            conserve_sz=config["conserve_qns"] && config["conserve_sz"],
        )
    elseif spin == 1
        siteinds("Fermion", n_sites; conserve_qns=config["conserve_qns"])
    else
        error("Unsupported spin=$spin")
    end

    os = OpSum()
    for term in spec["terms"]
        os = add_term(os, complex_coeff(term["coefficient"]), term["label"], spin)
    end

    H = MPO(os, sites)

    state = build_initial_state(
        n_sites,
        spin,
        n_occ,
        config["seed"],
        config["initial_state"],
    )

    psi0 = productMPS(sites, state)

    dmrg_time_t0 = time_ns()
    energy, psi = dmrg(
        H,
        psi0;
        nsweeps=config["nsweeps"],
        maxdim=config["maxdims"],
        cutoff=config["cutoff"],
    )
    dmrg_elapsed_s = (time_ns() - dmrg_time_t0) / 1e9
    dmrg_result = (energy=energy, psi=psi)

    observable_value = nothing
    if haskey(spec, "observable_terms")
        observable_os = OpSum()
        for term in spec["observable_terms"]
            observable_os = add_term(
                observable_os,
                complex_coeff(term["coefficient"]),
                term["label"],
                spin,
            )
        end
        O = MPO(observable_os, sites)
        observable_value = real(inner(dmrg_result.psi', O, dmrg_result.psi))
    end

    overall_elapsed_s = (time_ns() - overall_t0) / 1e9
    link_dims = collect_link_dims(dmrg_result.psi)
    output = Dict(
        "format" => "qbp_itensor_dmrg_result_v1",
        "model" => spec["model"],
        "lattice" => spec["lattice"],
        "spin" => spin,
        "n_sites" => n_sites,
        "n_occ" => n_occ,
        "model_params" => spec["model_params"],
        "num_terms" => length(spec["terms"]),
        "nsweeps" => config["nsweeps"],
        "maxdims" => config["maxdims"],
        "cutoff" => config["cutoff"],
        "seed" => config["seed"],
        "initial_state" => config["initial_state"],
        "energy" => dmrg_result.energy,
        "observable" => get(spec, "observable", "E"),
        "profile" => Dict(
            "dmrg" => Dict("elapsed_s" => dmrg_elapsed_s),
            "total" => Dict("elapsed_s" => overall_elapsed_s),
        ),
        "link_dims" => link_dims,
        "max_link_dim" => isempty(link_dims) ? 0 : maximum(link_dims),
        "avg_link_dim" => isempty(link_dims) ? 0.0 : sum(link_dims) / length(link_dims),
    )
    if !isnothing(observable_value)
        output["observable_value"] = observable_value
    end

    mkpath(dirname(config["output"]))
    open(config["output"], "w") do io
        JSON.print(io, output, 2)
    end
    println("DMRG energy: $(dmrg_result.energy)")
    if !isnothing(observable_value)
        println("DMRG ", spec["observable"], ": ", observable_value)
    end
    println("Wrote $(config["output"])")
end

main()
