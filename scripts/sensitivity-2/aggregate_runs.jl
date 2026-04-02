using CSV
using DataFrames
using Statistics: mean

EXPERIMENT = "sensitivity-2"
PARAMS = [:w, :inv_t, :a_mho]

function load_result(path::String)
    CSV.read(path, DataFrame)
end

function merge_results(path::String)
    files = readdir(path;join = true)
    filter!(endswith(".csv"), files)
    all = vcat(map(load_result, files)...)
    return all
end

function aggregate_results(param)
    BASE_PATH = "/spaths/experiments/$(EXPERIMENT)/$(param)"
    RUN_PATH = "$(BASE_PATH)/NOTICE"
    all = merge_results(RUN_PATH)
    return all
end


function main()
    dfs = DataFrame[]
    for param = PARAMS
        push!(dfs, aggregate_results(param))
    end
    df = vcat(dfs...)
    OUT_PATH = "/spaths/experiments/$(EXPERIMENT)/aggregate.csv"
    CSV.write(OUT_PATH, df)
end

main()
