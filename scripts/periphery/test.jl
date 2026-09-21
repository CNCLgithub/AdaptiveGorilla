################################################################################
# Script to run models on the Target-Ensemble Experiment (Study 2)
#
# Output is stored under `spaths/experiments/`
# See `README` for more information.
################################################################################


################################################################################
# Includes
################################################################################

using Gen
using CSV
using ArgParse
using DataFrames
using ProgressMeter
using AdaptiveGorilla
using AdaptiveGorilla: count_collisions
using LinearAlgebra: norm
import AdaptiveGorilla as AG

using Random
#Random.seed!()

################################################################################
# Command Line Interface
################################################################################

MODEL_VARIANTS = Dict(
    :mo => "Multi-Granular Optimization",
    :ja => "Just Attention",
)

ANALYSES_VARIANTS = [:NOTICE, :PERF]

s = ArgParseSettings()

@add_arg_table! s begin

    "--analyses"
    help = "Model analyses. Either NOTICE or PERF"
    range_tester = in(ANALYSES_VARIANTS)
    default = :NOTICE

    "model"
    help = "Model Variant"
    arg_type = Symbol
    range_tester = in(keys(MODEL_VARIANTS))
    default = :ja

    "scene"
    help = "Which scene to run"
    arg_type = Int64
    default = 2
end

PARAMS = parse_args(ARGS, s)

################################################################################
# Model Parameters
################################################################################

MODEL = PARAMS["model"]
MODEL_PARAMS = "/project/scripts/params/$(MODEL).toml"

# World model parameters; See "?InertiaWM" for documentation.
WM = load_wm_from_toml("/project/scripts/params/wm.toml")

################################################################################
# General Experiment Parameters
################################################################################

# which dataset to run
DATASET = "study2"
DPATH   = "/spaths/datasets/$(DATASET)/dataset.json"
SCENE   = PARAMS["scene"]
FRAMES  = 240

LONE_PARENT = false
SWAP_COLORS = false

################################################################################
# ANALYSES
################################################################################

ANALYSIS = PARAMS["analyses"]

if ANALYSIS == :NOTICE
    SHOW_GORILLA=true

elseif ANALYSIS == :PERF
    SHOW_GORILLA=false
end

################################################################################
# Analysis Parameters
################################################################################

RENDER = true
# RENDER = false

# Number of model runs per condition
CHAINS = RENDER ? 1 : 16

# The probability lower bound of gorilla noticing.
# The probability is implemented with `detect_gorilla` and it's marginal is
# estimated across the hyper particles.
# Pr(detect_gorilla) = 0.1 denotes a 10% confidence that the gorilla is present
# at a given moment in time (i.e., a frame)
NOTICE_P_THRESH = 0.20

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

function run_model!(pbar, exp)
    # Initializes the agent
    # (Done from scratch each time to avoid bugs / memory leaks)
    agent = load_agent(MODEL_PARAMS, exp.init_query)

    results = DataFrame(
        :frame => Int64[],
        :gorilla_p => Float64[],
        :collision_p => Float64[],
        :birth_p => Float64[],
        :distance => Vector{Union{Missing, Float64}}(undef, 0),
    )

    for t = 1:(FRAMES - 1)
        # println("###########                     ###########")
        # println("###########       TIME $(t)     ###########")
        # println("###########                     ###########")
        _results = test_agent!(agent, exp, t)
        distance = distance_to_centroid(exp, agent, t)
        _results[:frame] = t
        # _results[:distance] = distance
        push!(results, (;_results..., distance = distance))
        next!(pbar)
    end
    return results
end

################################################################################
# Main Entry
################################################################################

function main()
    result = NamedTuple[]
    pbar = Progress(
        (FRAMES-1);
        desc="Running $(MODEL) model...", dt = 1.0)
    experiment = TEnsExp(DPATH, WM, SCENE, SWAP_COLORS, LONE_PARENT, FRAMES;
                         show_gorilla = SHOW_GORILLA)
    gt_count = count_collisions(experiment)
    @show gt_count

    results = run_model!(pbar, experiment)
    show(results; allrows=true)
    finish!(pbar)
    return nothing
end;

main();
