using CSV
using DataFrames
using Statistics: mean

NOTICE_MIN_FRAMES = 18
DATASET = "target_ensemble/2025-06-09_W96KtK"
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
    BASE_PATH = "/spaths/experiments/$(DATASET)/$(model)-NOTICE"
    RUN_PATH = "$(BASE_PATH)/scenes"
    all = merge_results(RUN_PATH)
    g = groupby(all, [:scene, :color, :parent])
    c = combine(g,
                :ndetected =>
                    (x -> mean(>(NOTICE_MIN_FRAMES), x)) =>
                    :noticed,
        :count_error => mean)
    show(c; allrows=true)
    c[!, :model] .= model
    return c
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
