using CSV
using DataFrames

EXP = "periphery"

function load_result(path::String)
    CSV.read(path, DataFrame)
end

function merge_results(path::String)
    files = readdir(path;join = true)
    filter!(endswith(".csv"), files)
    all = vcat(map(load_result, files)...)
    return all
end

function aggregate_results()
    BASE_PATH = "/spaths/experiments/$(EXP)/run"
    RUN_PATH = "$(BASE_PATH)/runs"
    merge_results(RUN_PATH)
end


function main()
    df = aggregate_results()
    OUT_PATH = "/spaths/experiments/$(EXP)/aggregate.csv"
    CSV.write(OUT_PATH, df)
end

main()
