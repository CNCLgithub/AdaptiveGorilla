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
using ArgParse
using Gen_Compose
using ProgressMeter
using DataFrames, CSV
using AdaptiveGorilla

using AdaptiveGorilla: S3V, count_collisions, get_last_state, predict
using Distances: WeightedEuclidean

################################################################################
# Command Line Interface
################################################################################

MODEL_VARIANTS = Dict(:mo => "Multi-Granular Optimization",
                      :ta => "Task-Agnostic Regranularization",
                      :ja => "Just Attention",
                      :fr => "Fixed Resource")

ANALYSES_VARIANTS = [:NOTICE, :PERF]

s = ArgParseSettings()

@add_arg_table! s begin

    "--restart", "-r"
    help = "Whether to resume inference"
    action = :store_true

    "--analyses"
    help = "Model analyses. Either NOTICE or PERF"
    range_tester = in(ANALYSES_VARIANTS)
    default = :NOTICE

    "--nchains", "-n"
    help = "The number of chains to run"
    arg_type = Int
    default = 1

    "model"
    help = "Model Variant"
    arg_type = Symbol
    range_tester = in(keys(MODEL_VARIANTS))
    default = :mo

    "scene"
    help = "Which scene to run"
    arg_type = Int64
    default = 4
end

PARAMS = parse_args(ARGS, s)

################################################################################
# Model Parameters
################################################################################

MODEL = PARAMS["model"]
MODEL_PARAMS = "$(@__DIR__)/models/$(MODEL).toml"

# World model parameters; See "?InertiaWM" for documentation.
WM = load_wm_from_toml("$(@__DIR__)/models/wm.toml")

################################################################################
# General Experiment Parameters
################################################################################

# which dataset to run
DATASET = "target_ensemble/2025-06-09_W96KtK"
DPATH   = "/spaths/datasets/$(DATASET)/dataset.json"
SCENE   = PARAMS["scene"]
FRAMES  = 240

# 4 Conditions total: 2 colors x 2 gorilla parents
# LONE_PARENT = [true, false]
# SWAP_COLORS = [false, true]
# LONE_PARENT = [false]
LONE_PARENT = [false]
SWAP_COLORS = [false]

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

# Number of model runs per condition
CHAINS = PARAMS["nchains"]

# The probability lower bound of gorilla noticing.
# The probability is implemented with `detect_gorilla` and it's marginal is
# estimated across the hyper particles.
# Pr(detect_gorilla) = 0.1 denotes a 10% confidence that the gorilla is present
# at a given moment in time (i.e., a frame)
NOTICE_P_THRESH = 0.25

################################################################################
# Methods
################################################################################

function run_model!(pbar, exp)
    # Initializes the agent
    # (Done from scratch each time to avoid bugs / memory leaks)
    agent = load_agent(MODEL_PARAMS, exp.init_query)
    display(agent.memory)
    out = "/spaths/tests/target-ensemble"
    isdir(out) || mkpath(out)

    results = DataFrame(
        :frame => Int64[],
        :gorilla_p => Float64[],
        :collision_p => Float64[],
        :birth_p => Float64[],
    )
    for t = 1:(FRAMES - 1)
        # println("###########                     ###########")
        # println("###########       TIME $(t)     ###########")
        # println("###########                     ###########")
        _results = test_agent!(agent, exp, t)
        _results[:frame] = t
        push!(results, _results)
        render_agent_state(exp, agent, t, out)
        next!(pbar)
    end

    # Dissect internal of perception after last frame
    visp, visstate = mparse(agent.perception)
    for i = 1:visp.h
        chain = visstate.chains[i]
        particles = chain.state
        mll = AdaptiveGorilla.marginal_ll(particles.traces[1])
        mll = zeros(length(mll))
        for x = particles.traces
            mll .+= AdaptiveGorilla.marginal_ll(x)
        end
        mll .*= 1.0 / length(particles.traces)
        @show mll
    end
    return results
end

################################################################################
# Main Entry
################################################################################

function main()
    result = NamedTuple[]
    pbar = Progress(
        length(SWAP_COLORS) * length(LONE_PARENT) * CHAINS * (FRAMES-1);
        desc="Running $(MODEL) model...", dt = 1.0)
    for swap = SWAP_COLORS, lone = LONE_PARENT
        experiment = TEnsExp(DPATH, WM, SCENE, swap, lone, FRAMES)
        gt_count = count_collisions(experiment)
        @show gt_count
        results = run_model!(pbar, experiment)
        show(results; allrows=true)
        println()
        @show count(results[!, :gorilla_p] .> NOTICE_P_THRESH)
    end
    finish!(pbar)
    return nothing
end;

main();
