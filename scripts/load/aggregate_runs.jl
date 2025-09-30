using CSV
using DataFrames
using Statistics: mean


NOTICE_MIN_FRAMES = 18

DATASET = "load_curve"

MODELS = [:MO, :AC, :FR]

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
    model_results = merge_results(RUN_PATH)
    model_results[:, :model] .= model
    return model_results
end

function main()
    OUT_PATH = "/spaths/experiments/$(DATASET)/aggregate.csv"
    results = []
    for model = MODELS
        push!(results, aggregate_results(model))
    end
    all = vcat(results...)
    g = groupby(all, [:model, :scene, :color])
    c = combine(g,
                :ndetected =>
                    (x -> mean(>(NOTICE_MIN_FRAMES), x)) =>
                    :noticed,
		:count_error => mean => :error)
    show(c; allrows=true)
    CSV.write(OUT_PATH, c)
end

main()
