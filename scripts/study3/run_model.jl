################################################################################
# Script to run models on Study 3 - load curve
#
# Output is stored under `spaths/experiments/`
# See `README` for more information.
################################################################################


################################################################################
# Includes
################################################################################

using CSV
using Gen
using Random
using ArgParse
using DataFrames
using Gen_Compose
using ProgressMeter
using Statistics: mean, std

using AdaptiveGorilla
using AdaptiveGorilla: count_collisions


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

    "--analyses"
    help = "Model analyses. Either NOTICE or PERF"
    range_tester = in(ANALYSES_VARIANTS)
    default = :NOTICE

    "--nchains", "-n"
    help = "The number of chains to run"
    arg_type = Int
    default = 8

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

# Number of model runs per condition
CHAINS = PARAMS["nchains"]


################################################################################
# General Experiment Parameters
################################################################################

# Setting seed for reproducibility
Random.seed!(123)

# which dataset to run
DATASET = "load_curve"
DPATH   = "/spaths/datasets/$(DATASET)/dataset.json"
SCENE   = PARAMS["scene"]
FRAMES  = 240

# Number of targets and distractors
NTARGETS = 4
NDISTRACTORS = collect(3:8)
# Each condition is a distractor count
NCOND = length(NDISTRACTORS)


################################################################################
# Methods
################################################################################

# Run the model once, returns expected collision count
function run_model!(pbar, exp)
    # Initializes the agent
    # (Done from scratch each time to avoid bugs / memory leaks)
    agent = load_agent(MODEL_PARAMS, exp.init_query)
    collision_counts = 0.0
    elapsed = 0.0
    for t = 1:(FRAMES - 1)
        _results = test_agent!(agent, exp, t)
        collision_counts = _results[:collision_p]
        elapsed += _results[:time]
        next!(pbar)
    end
    collision_counts, elapsed
end

# Stores data from a single model run
RunSummary = @NamedTuple begin
    scene          :: Int64
    ndark          :: Int64
    chain          :: Int64
    gt_count       :: Int64
    expected_count :: Float64
    count_error    :: Float64
    time           :: Float64
end


################################################################################
# Main Entry
################################################################################

function main()
    nruns = NCOND * CHAINS
    nsteps = nruns * (FRAMES-1)
    pbar = Progress(nsteps;
                    desc="[Study 3] $(MODEL_VARIANTS[MODEL]) (x$(nruns))",
                    dt = 2.0)
    # Preallocate simulation results
    summaries = Vector{RunSummary}(undef, nruns)
    linds = LinearIndices((CHAINS, NCOND))

    # Go through each of the conditions
    for (i, ndistractor) = enumerate(NDISTRACTORS)
        # Load the world model
        wm = load_wm_from_toml("$(@__DIR__)/params/wm.toml";
                               object_rate = Float64(NTARGETS + ndistractor))
        # Load the experiment
        experiment = LoadCurve(wm, DPATH, SCENE, FRAMES, NTARGETS, ndistractor)
        # Retrieve the number of true collisions
        gt_count = count_collisions(experiment)
        # Run the model several chains
        Threads.@threads for c = 1:CHAINS
            
            expected_count, elapsed = run_model!(pbar, experiment)
            count_error = abs(gt_count - expected_count) / gt_count

            summaries[linds[c, i]] = RunSummary((
                scene          = SCENE,
                ndark          = ndistractor,
                chain          = c,
                gt_count       = gt_count,
                expected_count = expected_count,
                count_error    = count_error,
                time           = elapsed,
            ))
        end
    end
    finish!(pbar)
    out_dir = "/spaths/experiments/$(DATASET)/$(MODEL)-$(ANALYSIS)/scenes"
    isdir(out_dir) || mkpath(out_dir)
    df = DataFrame(summaries)
    CSV.write("$(out_dir)/$(SCENE).csv", df)

    # Quick display
    display(combine(groupby(df, [:scene, :ndark]),
                    :gt_count => mean,
                    :expected_count => mean,
                    :count_error => mean,
                    :count_error => std,
                    :time => mean))
    return nothing
end;

main();
