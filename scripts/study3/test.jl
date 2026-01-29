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
using AdaptiveGorilla: count_collisions

################################################################################
# Command Line Interface
################################################################################

MODEL_VARIANTS = Dict(:mo => "Multi-Granular Optimization",
                      :ta => "Task-Agnostic Regranularization",
                      :ja => "Just Attention",
                      :fr => "Fixed Resource")

s = ArgParseSettings()

@add_arg_table! s begin

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
MODEL_PARAMS = "$(@__DIR__)/params/$(MODEL).toml"


################################################################################
# General Experiment Parameters
################################################################################

# which dataset to run
DATASET = "study3"
DPATH   = "/spaths/datasets/$(DATASET)/dataset.json"
SCENE   = PARAMS["scene"]
FRAMES  = 240

NTARGETS = 4
NDISTRACTORS = 8

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
NOTICE_P_THRESH = 0.5

################################################################################
# Methods
################################################################################

function run_model!(pbar, exp)
    out = "/spaths/tests/load"
    isdir(out) || mkpath(out)
    # Initializes the agent
    # (Done from scratch each time to avoid bugs / memory leaks)
    agent = load_agent(MODEL_PARAMS, exp.init_query)
    results = DataFrame(
        :frame => Int64[],
        :collision_p => Float64[],
        :time => Float64[],
    )
    for t = 1:(FRAMES - 1)
        _results = test_agent!(agent, exp, t)
        _results[:frame] = t
        push!(results, _results)
        render_agent_state(exp, agent, t, out)
        next!(pbar)
    end
    return results
end

################################################################################
# Main Entry
################################################################################

function main()
    result = NamedTuple[]
    nsteps = FRAMES-1
    pbar = Progress(nsteps; desc="Running $(MODEL) model...", dt = 1.0)
    # Load the world model
    wm = load_wm_from_toml("$(@__DIR__)/params/wm.toml";
                            object_rate = Float64(NTARGETS + NDISTRACTORS))
    # Load the experiment
    experiment = LoadCurve(wm, DPATH, SCENE, FRAMES, NTARGETS, NDISTRACTORS)
    # Retrieve the number of true collisions
    gt_count = count_collisions(experiment)
    @show gt_count
    results = run_model!(pbar, experiment)
    display(last(results))
    @show sum(results[!, :time])
    # show(results; allrows=true)
    println()
    finish!(pbar)
    return nothing
end;

main();
