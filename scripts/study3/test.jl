################################################################################
# Script to run models on the Load Experiment (Study 1)
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

using Profile
using StatProfilerHTML

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
    default = :fr

    "scene"
    help = "Which scene to run"
    arg_type = Int64
    default = 3
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
NDISTRACTORS = 6

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
        # @profile _results = test_agent!(agent, exp, t)
        _results[:frame] = t
        push!(results, _results)
        # render_agent_state(exp, agent, t, out)
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

    Profile.clear()
    results = run_model!(pbar, experiment)
    # statprofilehtml()
    # display(last(results))
    show(results; allrows=true)
    println()
    @show sum(results[!, :time])
    println()
    finish!(pbar)
    @show gt_count
    return nothing
end;

main();
