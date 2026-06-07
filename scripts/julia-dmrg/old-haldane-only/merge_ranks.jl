#!/usr/bin/env julia

using Printf

if !(1 <= length(ARGS) <= 2)
    println("Usage: julia scripts/julia-dmrg/old-haldane-only/merge_ranks.jl <input_dir> [output_file]")
    exit(1)
end

input_dir = abspath(ARGS[1])
isdir(input_dir) || error("Input directory does not exist: $input_dir")

default_output = joinpath(input_dir, "merged.jsonl")
output_file = abspath(length(ARGS) >= 2 ? ARGS[2] : default_output)

rank_files = sort(
    filter(name -> occursin(r"^rank\d+_of_\d+_.*\.jsonl$", name), readdir(input_dir))
)

isempty(rank_files) && error("No rank JSONL files found in: $input_dir")

output_name = basename(output_file)
rank_files = filter(name -> name != output_name, rank_files)

isempty(rank_files) && error("No input rank JSONL files remain after excluding output file: $output_file")

mkpath(dirname(output_file))

line_count = Ref(0)
open(output_file, "w") do out_io
    for filename in rank_files
        filepath = joinpath(input_dir, filename)
        open(filepath, "r") do in_io
            for line in eachline(in_io; keep=true)
                isempty(strip(line)) && continue
                write(out_io, line)
                endswith(line, '\n') || write(out_io, '\n')
                line_count[] += 1
            end
        end
    end
end

@printf("Merged %d files into %s (%d records)\n", length(rank_files), output_file, line_count[])
