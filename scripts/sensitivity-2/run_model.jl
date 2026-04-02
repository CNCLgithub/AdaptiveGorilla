################################################################################
# Script to run sensitivity analyses for Study 2
#
# Output is stored under `spaths/experiments/`
# See `README` for more information.
################################################################################


################################################################################
# Includes
################################################################################

using CSV
using TOML
using Random
using ArgParse
using DataFrames
using ProgressMeter
using AdaptiveGorilla
using Statistics: mean
using UnicodePlots: Plot, lineplot!, histogram


################################################################################
# Command Line Interface
################################################################################

PARAM_VARIANTS = Dict(:w => "MLL weight",
                      :inv_t => "Task Exergy inv. temp",
                      :a_mho => "Task Anergy sensitivity")

PARAM_RANGES = Dict(:w => (400.0, 600.0),
                    :inv_t => (.01, .20),
                    :a_mho => (20.0, 40.0))
PARAMS_BASE = ["Memory" , "params" , "fitness" , "params"]
PARAM_PATHS = Dict(:w => "mll_beta" ,
                   :inv_t => "tenergy_inv_temp",
                   :a_mho => "complexity_factor")


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

    "param"
    help = "MO parameter to test"
    arg_type = Symbol
    range_tester = in(keys(PARAM_VARIANTS))
    default = :a_mho

    "scene"
    help = "Which scene to run"
    arg_type = Int64
    default = 1
end

PARAMS = parse_args(ARGS, s)

################################################################################
# Model Parameters
################################################################################

MODEL = :mo
MODEL_PARAMS = "$(@__DIR__)/params/mo.toml"

MODEL_PARAM_KEY = PARAMS["param"]
MODEL_PARAM_PATH = PARAM_PATHS[MODEL_PARAM_KEY]
MODEL_PARAM_LOW_HIGH = PARAM_RANGES[MODEL_PARAM_KEY]

WM = load_wm_from_toml("$(@__DIR__)/params/wm.toml")

function configure_params(param_val::Float64)
    # load original TOML
    head = toml = TOML.parsefile(MODEL_PARAMS)
    # Retrieve parameter address and configure it
    for step = PARAMS_BASE
       head = head[step]
    end
    head[MODEL_PARAM_PATH] = param_val
    return toml
end

################################################################################
# General Experiment Parameters
################################################################################

# Setting seed for reproducibility
Random.seed!(123)

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

function run_model!(pbar, experiment, param_val)
    # Initializes the agent
    params = configure_params(param_val)
    agent = load_agent(params, experiment.init_query)
    colp = 0.0
    noticed = 0
    pgorilla = Vector{Float64}(undef, FRAMES-1)
    for t = 1:(FRAMES - 1)
        _results = test_agent!(agent, experiment, t)
        colp = _results[:collision_p]
        if  _results[:gorilla_p] > NOTICE_P_THRESH
            noticed += 1
        end
        pgorilla[t] = _results[:gorilla_p]
        next!(pbar)
    end
    (noticed, pgorilla, colp)
end

RunSummary = @NamedTuple begin
    param          :: Symbol
    param_val      :: Float64
    scene          :: Int64
    color          :: Symbol
    parent         :: Symbol
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
    nruns = 2 * NSC * NP * CHAINS
    nsteps = nruns * (FRAMES-1)
    pbar = Progress(nsteps;
                    desc="Running $(MODEL) model...",
                    dt = 1.0)
    # Preallocate simulation results
    summaries = Vector{RunSummary}(undef, nruns)
    linds = LinearIndices((CHAINS, NP, NSC, 2))
    # Go through each of the conditions
    for (i, swap) = enumerate(SWAP_COLORS),
        (j, lone) = enumerate(LONE_PARENT),
        (k, param_val ) = enumerate(MODEL_PARAM_LOW_HIGH)

        color = swap ? :dark : :light
        parent = lone ? :lone : :grouped
        # Load the experiment
        experiment = TEnsExp(DPATH, WM, SCENE, swap, lone, FRAMES,
                             show_gorilla=SHOW_GORILLA)

        # Retrieve the number of true collisions
        gt_count = count_collisions(experiment)

        # Run the model several chains
        Threads.@threads for c = 1:CHAINS
            run = @timed run_model!(pbar, experiment, param_val)
            ndetected, pnoticed, expected_count = run.value
            count_error = abs(gt_count - expected_count) / gt_count

            summaries[linds[c,k,j,i]] = RunSummary((
                param          = MODEL_PARAM_KEY,
                param_val      = param_val,
                scene          = SCENE,
                color          = color,
                parent         = parent,
                chain          = c,
                ndetected      = ndetected,
                expected_count = expected_count,
                count_error    = count_error,
                time           = run.time
            ))

        end
    end
    finish!(pbar)

    # Record results to CSV
    out_dir = "/spaths/experiments/sensitivity-2/$(MODEL_PARAM_KEY)/$(ANALYSIS)"
    isdir(out_dir) || mkpath(out_dir)
    df = DataFrame(summaries)
    CSV.write("$(out_dir)/$(SCENE).csv", df)

    ## Additional visualizations
    count_f = x -> count(>=(24), x) / CHAINS
    by_cond = groupby(df, [:param_val, :color, :parent])
    display(combine(by_cond, :ndetected => count_f))

    return nothing
end;

main();
