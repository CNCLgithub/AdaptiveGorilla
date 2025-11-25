using CSV
using DataFrames
using Statistics: mean


NOTICE_MIN_FRAMES = 18

DATASET = "target_ensemble/2025-06-09_W96KtK"

MODEL = :FR
BASE_PATH = "/spaths/experiments/$(DATASET)/$(MODEL)-NOTICE"
RUN_PATH = "$(BASE_PATH)/scenes"
OUT_PATH = "$(BASE_PATH)/aggregate.csv"

function load_result(path::String)
    CSV.read(path, DataFrame)
end

function merge_results(path::String)
    files = readdir(path;join = true)
    filter!(endswith(".csv"), files)
    all = vcat(map(load_result, files)...)
    return all
end


function main()
    all = merge_results(RUN_PATH)
    g = groupby(all, [:scene, :color, :parent])
    c = combine(g,
                :ndetected =>
                    (x -> mean(>(NOTICE_MIN_FRAMES), x)) =>
                    :noticed,
        :error => mean)
    show(c; allrows=true)
    CSV.write(OUT_PATH, c)
    c = combine(all,
        :error => mean)
    show(c; allrows=true)
end

main()
