################################################################################
# Script to run models on Study 3 - load curve
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
using Statistics: mean

using AdaptiveGorilla
using AdaptiveGorilla: S3V, count_collisions

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
    default = :fr

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
DATASET = "load_curve"
DPATH   = "/spaths/datasets/$(DATASET)/dataset.json"
SCENE   = PARAMS["scene"]
FRAMES  = 240

NTARGETS = 4
NDISTRACTORS = collect(3:8)
NCOND = length(NDISTRACTORS)

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

# Stores data from a single model run
RunSummary = @NamedTuple begin
    scene          :: Int64
    ndark          :: Int64
    chain          :: Int64
    ndetected      :: Int64
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
            
            run = @timed run_model!(pbar, experiment)
            ndetected, expected_count = run.value
            count_error = abs(gt_count - expected_count) / gt_count

            summaries[linds[c, i]] = RunSummary((
                scene          = SCENE,
                ndark          = ndistractor,
                chain          = c,
                ndetected      = ndetected,
                expected_count = expected_count,
                count_error    = count_error,
                time           = run.time
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
                    :expected_count => mean,
                    :count_error => mean,
                    :time => mean))
    return nothing
end;

main();
