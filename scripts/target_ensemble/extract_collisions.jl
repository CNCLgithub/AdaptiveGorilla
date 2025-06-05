using CSV
using JSON3
using MOTCore
using DataFrames
using StaticArrays
using LinearAlgebra: norm, cross, dot

const S2V = SVector{2, Float64}

# out of bounds
function oob(radius::Float64, x::Float64, y::Float64,
             w::Float64, h::Float64)
    dx = abs(x + sign(x)*radius)
    dy = abs(y + sign(y)*radius)
    (dx >= 0.5 * w) || (dy >= 0.5 * h)
end

function count_collisions(states, r::Float64,
                          w::Float64, h::Float64)
    nobj = length(states[1])
    nframes = length(states)
    # obj_pos = Vector{S2V}(undef, nobj)
    col_frame = fill(-Inf, 4)
    count = 0
    # initialize
    # @inbounds for i = 1:nobj
    #     x, y = states[1][i]
    #     obj_pos[i] = S2V(x, y)
    # end
    # iterate through time steps
    max_x = 0.0
    max_y = 0.0
    i = 4
    for t = 1:nframes
        # for i = 1:4
            x, y = states[t][i]
            last_col = col_frame[i]
            if oob(r, x, y, w, h) && (t - last_col) > 2
                println("FRAME: $(t)")
                count += 1
                col_frame[i] = t
            end
            max_x = max(abs(x), max_x)
            max_y = max(abs(y), max_y)
        # end
    end
    return count
end

function load_dataset(path::String)
    local trials
    open(path, "r") do io
        trials = JSON3.read(io)["trials"]
    end
    return trials
end

function main()

    dataset = "target_ensemble"
    version = "0.1"

    data_dir = "/spaths/datasets/$(dataset)"
    out_dir = "$(data_dir)/$(version)"
    dpath = "$(out_dir)/dataset.json"
    outpath = "$(out_dir)/collision_counts.csv"

    trials = load_dataset(dpath)
    ntrials = length(trials)
    counts = zeros(Int64, ntrials)
    for i = 1:ntrials
        println("TRIAL: $(i)")
        states = trials[i]["positions"]
        counts[i] =
            count_collisions(states, 10.0, 720.0, 480.0)
    end
    results = DataFrame(trial = 1:ntrials,
                        counts = counts)
    display(results)
    CSV.write(outpath, results)
end


main();
