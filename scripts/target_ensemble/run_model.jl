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

using AdaptiveGorilla: S3V, count_collisions
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
    default = 32

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
LONE_PARENT = [true, false]
# SWAP_COLORS = [false, true]
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
NOTICE_P_THRESH = 0.5

################################################################################
# Methods
################################################################################

function run_model!(pbar, exp)
    # Initializes the agent
    # (Done from scratch each time to avoid bugs / memory leaks)
    agent = load_agent(MODEL_PARAMS, exp.init_query)
    colp = 0.0
    noticed = 0
    for t = 1:(FRAMES - 1)
        _results = test_agent!(agent, exp, t)
        colp = _results[:collision_p]
        if  _results[:gorilla_p] > NOTICE_P_THRESH
            noticed += 1
        end
        next!(pbar)
    end
    (noticed, colp)
end

################################################################################
# Main Entry
################################################################################

function main()
    result = NamedTuple[]
    nsteps = length(SWAP_COLORS) * length(LONE_PARENT) * CHAINS * (FRAMES-1)
    pbar = Progress(nsteps; desc="Running $(MODEL) model...", dt = 1.0)
    # Go through each of the conditions
    for swap = SWAP_COLORS, lone = LONE_PARENT
        # Load the experiment
        experiment = TEnsExp(DPATH, WM, SCENE, swap, lone, FRAMES)
        # Retrieve the number of true collisions
        gt_count = count_collisions(experiment)
        # Run the model several chains
        Threads.@threads for c = 1:CHAINS
            ndetected, expected_count = run_model!(pbar, experiment)
            count_error = abs(gt_count - expected_count) / gt_count
            push!(result,
                  (scene          = SCENE,
                   color          = swap ? :dark : :light,
                   parent         = lone ? :lone : :grouped,
                   chain          = c,
                   ndetected      = ndetected,
                   expected_count = expected_count,
                   count_error    = count_error))
        end
    end
    finish!(pbar)
    out_dir = "/spaths/experiments/$(DATASET)/$(MODEL)-$(ANALYSIS)/scenes"
    isdir(out_dir) || mkpath(out_dir)
    df = DataFrame(result)
    count_f = x -> count(>(24.0), x) / CHAINS
    display(combine(groupby(df, [:scene, :color, :parent]), :ndetected => count_f))
    CSV.write("$(out_dir)/$(SCENE).csv", df)
    return nothing
end;

main();
