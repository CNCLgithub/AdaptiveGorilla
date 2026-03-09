using CSV
using DataFrames
using Statistics: mean

DATASET = "study3"
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
    RUN_PATH = "/spaths/experiments/$(DATASET)/$(model)/runs"
    all = merge_results(RUN_PATH)
    g = groupby(all, [:scene, :ndark])
    c = combine(g,
        :time => mean,
        :count_error => mean)
    show(c; allrows=true)
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
