################################################################################
# Script to run models on the Target-Ensemble Experiment (Study 2)
#
# Output is stored under `spaths/experiments/`
# See `README` for more information.
################################################################################


################################################################################
# Includes
################################################################################

using CSV
using Random
using ArgParse
using DataFrames
using ProgressMeter
using Gen: has_value
using AdaptiveGorilla
using Statistics: mean
using LinearAlgebra: norm
import AdaptiveGorilla as AG

################################################################################
# Command Line Interface
################################################################################

MODEL_VARIANTS = Dict(
    :mo => "Multi-Granularity Optimization",
    :ja => "Just Attention",
)

ANALYSES_VARIANTS = [:NOTICE, :PERF]

s = ArgParseSettings()
@add_arg_table! s begin

    "--analyses"
    help = "Model analyses. Either NOTICE or PERF"
    range_tester = in(ANALYSES_VARIANTS)
    default = :NOTICE

    "--nchains", "-n"
    help = "The number of chains to run"
    arg_type = Int
    default = 64

    "model"
    help = "Model Variant"
    arg_type = Symbol
    range_tester = in(keys(MODEL_VARIANTS))
    default = :ja

    "scene"
    help = "Which scene to run"
    arg_type = Int64
    default = 1
end

PARAMS = parse_args(ARGS, s)

################################################################################
# Model Parameters
################################################################################

MODEL = PARAMS["model"]
MODEL_PARAMS = "/project/scripts/params/$(MODEL).toml"

WM = load_wm_from_toml("/project/scripts/params/wm.toml")


################################################################################
# General Experiment Parameters
################################################################################

# Setting seed for reproducibility
Random.seed!(321)

# which dataset to run
DATASET = "study2"
DPATH   = "/spaths/datasets/$(DATASET)/dataset.json"
SCENE   = PARAMS["scene"]
FRAMES  = 240

# 4 Conditions total: 2 colors x 2 gorilla parents

# Gorilla parent
LONE_PARENT = [true, false]
NP = length(LONE_PARENT)

# Swapping all object colors
SWAP_COLORS = [false, true]
NSC = length(SWAP_COLORS)


################################################################################
# Analysis Parameters
################################################################################

# Number of model runs per condition
CHAINS = PARAMS["nchains"]

# The probability lower bound of gorilla noticing.
# The probability is implemented with `detect_gorilla` and it's marginal is
# estimated across the hyper particles.
# Pr(detect_gorilla) = 0.1 denotes a 10% confidence that the gorilla is present
# at a given moment in time (i.e., a frame)
NOTICE_P_THRESH = 0.2

ANALYSIS = PARAMS["analyses"]

if ANALYSIS == :NOTICE
    SHOW_GORILLA=true

elseif ANALYSIS == :PERF
    SHOW_GORILLA=false
end

################################################################################
# Methods
################################################################################

# NOTE: assumes agent has adaptive computation
function attention_centroid(agent)
    prot, state = mparse(agent.attention)
    xs = Array(state.dPi.coords)
    ys = Array(state.dPi.samples)
    ws = softmax(ys)  
    mu = sum(xs .* ws)
    AG.S2V(mu[1], mu[2])
end

function probe_point(exp, t)
    mask_id = 9 # exp.lone_parent ? 4 : 1
    masks = exp.observations[t]
    detection = masks[mask_id]
    AG.S2V(detection.x, detection.y)
end

function has_gorilla(exp, t)
    masks = exp.observations[t]
    mask_id = 9 # exp.lone_parent ? 4 : 1
    (has_value(masks, mask_id), masks)
end

function distance_to_centroid(exp, agent, t)
    valid, masks = has_gorilla(exp, t)
    valid || return missing
    detection = masks[9]
    p = probe_point(exp, t)
    c = attention_centroid(agent)
    norm(p - c)
end

function run_model!(pbar, experiment)
    agent = load_agent(MODEL_PARAMS, experiment.init_query)
    results = DataFrame(
        :frame => Int64[],
        :distance => Vector{Union{Missing, Float64}}(undef, 0),
    )
    for t = 1:(FRAMES - 1)
        test_agent!(agent, experiment, t)
        distance = distance_to_centroid(experiment, agent, t)
        push!(results, (;frame = t, distance = distance))
        next!(pbar)
    end
    mean(skipmissing(results[!, :distance]))
end

RunSummary = @NamedTuple begin
    scene          :: Int64
    color          :: Symbol
    parent         :: Symbol
    chain          :: Int64
    distance       :: Float64
end

################################################################################
# Main Entry
################################################################################

function main()
    nruns = NSC * NP * CHAINS
    nsteps = nruns * (FRAMES-1)
    pbar = Progress(nsteps;
                    desc="Running $(MODEL) model...",
                    dt = 1.0)
    # Preallocate simulation results
    summaries = Vector{RunSummary}(undef, nruns)
    # time_series = Vector{TimeSeries}(undef, nruns)
    linds = LinearIndices((CHAINS, NP, NSC))
    # Go through each of the conditions
    for (i, swap) = enumerate(SWAP_COLORS), (j, lone) = enumerate(LONE_PARENT)

        color = swap ? :dark : :light
        parent = lone ? :lone : :grouped
        # Load the experiment
        experiment = TEnsExp(DPATH, WM, SCENE, swap, lone, FRAMES,
                             show_gorilla=SHOW_GORILLA)
        # Retrieve the number of true collisions
        gt_count = count_collisions(experiment)
        # @show gt_count
        # Run the model several chains
        Threads.@threads for c = 1:CHAINS
            distance = run_model!(pbar, experiment)
            summaries[linds[c,j,i]] = RunSummary((
                scene          = SCENE,
                color          = color,
                parent         = parent,
                chain          = c,
                distance       = distance,
            ))
        end
    end
    finish!(pbar)

    # Record results to CSV
    out_dir = "/spaths/experiments/periphery/runs"
    isdir(out_dir) || mkpath(out_dir)
    df = DataFrame(summaries)
    CSV.write("$(out_dir)/$(SCENE).csv", df)

    return nothing
end;

main();
