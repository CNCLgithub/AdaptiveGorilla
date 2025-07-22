using CSV
using JSON3
using MOTCore
using DataFrames
using StaticArrays
using LinearAlgebra: norm, cross, dot

const S2V = SVector{2, Float64}

# The object indices for the lone
# and grouped gorilla parents
const LIDX = 4;
const GIDX = 1;

# Number of frames gorilla was present
const GORILLA_DUR = 60;

function load_dataset(path::String)
    local trials
    open(path, "r") do io
        trials = JSON3.read(io)["trials"]
    end
    return trials
end

"""
Determines the average L2 distance of
target to all other objects.
"""
function crowding(positions, target::Int)
    tx,ty = positions[target]
    n = length(positions)
    d = 0.0
    @inbounds for i = 1:n
        i == target && continue
        x, y = positions[i]
        d += sqrt((tx - x)^2 + (ty - y)^2)
    end
    d / n
end



function analyze_trial(data)
    gorilla = data["gorilla"]
    gorilla_start = gorilla["frame"]
    gorilla_stop  = gorilla_start + GORILLA_DUR

    positions = data["positions"]
    # The average crowding for lone and grouped
    lone_stat = 0.0
    group_stat = 0.0
    for t = gorilla_start:gorilla_stop
        step = positions[t]
        lone_stat += crowding(step, LIDX)
        group_stat += crowding(step, GIDX)
    end
    lone_stat *= 1.0 / GORILLA_DUR
    group_stat *= 1.0 / GORILLA_DUR

    return (lone_stat, group_stat)
end

function main()

    dataset = "target_ensemble"
    version = "2025-06-09_W96KtK"

    data_dir = "/spaths/datasets/$(dataset)"
    out_dir = "$(data_dir)/$(version)"
    dpath = "$(out_dir)/dataset.json"
    outpath = "$(out_dir)/crowding_stats.csv"

    trials = load_dataset(dpath)
    ntrials = length(trials)
    lone_stat  = Vector{Float64}(undef, ntrials)
    group_stat = Vector{Float64}(undef, ntrials)
    for i = 1:ntrials
        println("TRIAL: $(i)")
        trial = trials[i]
        lone_stat[i], group_stat[i] =
            analyze_trial(trial)
    end
    results = DataFrame(scene = 1:ntrials,
                        Alone = lone_stat,
                        Grouped = group_stat)
    display(results)
    CSV.write(outpath, results)
end


main();
