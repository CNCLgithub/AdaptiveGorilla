################################################################################
# Script to run models on the Target-Ensemble Experiment (Study 2)
#
# Output is stored under `spaths/experiments/`
# See `README` for more information.
################################################################################


################################################################################
# Includes
################################################################################

using ArgParse
using ProgressMeter
using DataFrames, CSV
using AdaptiveGorilla
using Statistics: mean
using UnicodePlots: Plot, lineplot!, histogram

################################################################################
# Command Line Interface
################################################################################

MODEL_VARIANTS = Dict(:mo => "Multi-Granularity Optimization",
                      :ta => "Task-Agnostic",
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
    default = 64

    "model"
    help = "Model Variant"
    arg_type = Symbol
    range_tester = in(keys(MODEL_VARIANTS))
    default = :mo

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
NOTICE_P_THRESH = 0.1

################################################################################
# Methods
################################################################################

function run_model!(pbar, exp)
    # Initializes the agent
    # (Done from scratch each time to avoid bugs / memory leaks)
    agent = load_agent(MODEL_PARAMS, exp.init_query)
    colp = 0.0
    noticed = 0
    pgorilla = Vector{Float64}(undef, FRAMES-1)
    for t = 1:(FRAMES - 1)
        _results = test_agent!(agent, exp, t)
        colp = _results[:collision_p]
        if  _results[:gorilla_p] > NOTICE_P_THRESH
            noticed += 1
        end
        pgorilla[t] = _results[:gorilla_p]
        next!(pbar)
    end
    (noticed, pgorilla, colp)
end

################################################################################
# Main Entry
################################################################################

function main()
    result = NamedTuple[]
    nsteps = length(SWAP_COLORS) * length(LONE_PARENT) * CHAINS * (FRAMES-1)
    pbar = Progress(nsteps; desc="Running $(MODEL) model...", dt = 1.0)
    noticed_df = DataFrame(;
                           color = Symbol[],
                           parent = Symbol[],
                           chain = Int64[],
                           frame = Int64[],
                           pnotice = Float64[])
    # Go through each of the conditions
    for swap = SWAP_COLORS, lone = LONE_PARENT
        color = swap ? :dark : :light
        # Load the experiment
        experiment = TEnsExp(DPATH, WM, SCENE, swap, lone, FRAMES)
        # Retrieve the number of true collisions
        gt_count = count_collisions(experiment)
        # Run the model several chains
        Threads.@threads for c = 1:CHAINS
            run = @timed run_model!(pbar, experiment)
            ndetected, pnoticed, expected_count = run.value
            count_error = abs(gt_count - expected_count) / gt_count
            push!(result,
                  (scene          = SCENE,
                   color          = color,
                   parent         = lone ? :lone : :grouped,
                   chain          = c,
                   ndetected      = ndetected,
                   expected_count = expected_count,
                   count_error    = count_error,
                   time           = run.time))
            append!(noticed_df,
                    DataFrame(color = color,
                              parent = lone ? :lone : :group,
                              chain = c,
                              frame = 1:(FRAMES-1),
                              pnotice = pnoticed))
        end
    end
    finish!(pbar)

    # Record results to CSV
    out_dir = "/spaths/experiments/" *
        "$(DATASET)/$(MODEL)-$(ANALYSIS)" *
        "/scenes"
    isdir(out_dir) || mkpath(out_dir)
    df = DataFrame(result)
    CSV.write("$(out_dir)/$(SCENE).csv", df)
    count_f = x -> count(>(6), x) / CHAINS

    by_cond = groupby(df, [:color, :parent])
    display(combine(by_cond, :ndetected => count_f))
    for k = keys(by_cond)
        g = by_cond[k]
        display(
            histogram(g[!, :ndetected], nbins=60, vertical=true, height=10,
                      title = repr(NamedTuple(k)))
        )
    end

    by_frame = combine(groupby(noticed_df, [:color, :parent, :frame]),
                       :pnotice => mean)
    plot = Plot(;
                xlabel = "t",
                ylabel = "Pr(Notice)",
                xlim = (1, FRAMES-1),
                ylim = (0, 1),
                title="Chain Averages"
                )
    g_by_frame = groupby(by_frame, [:color, :parent])
    for k = keys(g_by_frame)
        g = g_by_frame[k]
        lineplot!(plot, g[!, :frame], g[!, :pnotice_mean],
                  name = repr(NamedTuple(k)))
    end
    display(plot)


    return nothing
end;

main();
