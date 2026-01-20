using CSV
using DataFrames
using Statistics: mean


NOTICE_MIN_FRAMES = 18

DATASET = "study1"

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
    OUT_PATH = "$(BASE_PATH)/aggregate.csv"
    all = merge_results(RUN_PATH)
    g = groupby(all, [:color])
    c = combine(g,
                :ndetected =>
                    (x -> mean(>(NOTICE_MIN_FRAMES), x)) =>
                    :noticed,
        :count_error => mean)
    show(c; allrows=true)
    CSV.write(OUT_PATH, c)
end

function main()
    for model = MODELS
        println(model)
        aggregate_results(model)
        println("")
    end
end

main()
