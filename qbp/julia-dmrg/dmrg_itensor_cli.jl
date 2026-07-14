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
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end
    isnothing(config["hamiltonian"]) && error("--hamiltonian is required")
    isnothing(config["output"]) && error("--output is required")
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

function build_initial_state(n_sites::Int, spin::Int, n_occ::Int, seed::Int)
    rng = MersenneTwister(seed)
    if spin == 2
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

function profile_section!(f::Function, profile::Dict{String, Any}, label::String)
    t0 = time_ns()
    result = nothing
    alloc = @allocated begin
        result = f()
    end
    profile[label] = Dict(
        "elapsed_s" => (time_ns() - t0) / 1e9,
        "alloc_bytes" => Float64(alloc),
    )
    return result
end

function main()
    config = parse_args(ARGS)
    spec = JSON.parsefile(config["hamiltonian"])
    spin = Int(spec["spin"])
    n_sites = Int(spec["n_sites"])
    n_occ = Int(spec["n_occ"])
    profile = Dict{String, Any}()

    sites = profile_section!(profile, "siteinds") do
        if spin == 2
            siteinds("Electron", n_sites; conserve_qns=config["conserve_qns"])
        elseif spin == 1
            siteinds("Fermion", n_sites; conserve_qns=config["conserve_qns"])
        else
            error("Unsupported spin=$spin")
        end
    end

    os = profile_section!(profile, "build_opsum") do
        local_os = OpSum()
        for term in spec["terms"]
            local_os = add_term(local_os, complex_coeff(term["coefficient"]), term["label"], spin)
        end
        local_os
    end

    H = profile_section!(profile, "build_mpo") do
        MPO(os, sites)
    end

    state = profile_section!(profile, "build_init_state") do
        build_initial_state(n_sites, spin, n_occ, config["seed"])
    end

    psi0 = profile_section!(profile, "build_product_mps") do
        productMPS(sites, state)
    end

    dmrg_result = profile_section!(profile, "dmrg") do
        energy, psi = dmrg(
            H,
            psi0;
            nsweeps=config["nsweeps"],
            maxdim=config["maxdims"],
            cutoff=config["cutoff"],
        )
        (energy=energy, psi=psi)
    end

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
        "conserve_qns" => config["conserve_qns"],
        "energy" => dmrg_result.energy,
        "profile" => profile,
        "link_dims" => link_dims,
        "max_link_dim" => isempty(link_dims) ? 0 : maximum(link_dims),
        "avg_link_dim" => isempty(link_dims) ? 0.0 : sum(link_dims) / length(link_dims),
    )

    mkpath(dirname(config["output"]))
    open(config["output"], "w") do io
        JSON.print(io, output, 2)
    end
    println("DMRG energy: $(dmrg_result.energy)")
    println("Wrote $(config["output"])")
end

main()
