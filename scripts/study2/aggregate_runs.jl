using CSV
using DataFrames
using Statistics: mean

DATASET = "study2"
MODELS = [:mo, :ja, :ta, :fr]

function load_result(path::String)
    CSV.read(path, DataFrame)
end

function merge_results(path::String)
    files = readdir(path;join = true)
    filter!(endswith(".csv"), files)
    all = vcat(map(load_result, files)...)
    return all
end

function aggregate_results(model)
    BASE_PATH = "/spaths/experiments/$(DATASET)/$(model)"
    RUN_PATH = "$(BASE_PATH)/NOTICE"
    all = merge_results(RUN_PATH)
    all[!, :model] .= model
    return all
end


function main()
    dfs = DataFrame[]
    for model = MODELS
        println(model)
        push!(dfs, aggregate_results(model))
        println("")
    end
    df = vcat(dfs...)
    OUT_PATH = "/spaths/experiments/$(DATASET)/aggregate.csv"
    CSV.write(OUT_PATH, df)
end

main()
