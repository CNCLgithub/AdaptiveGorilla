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
using Gen_Compose
using ProgressMeter
using AdaptiveGorilla
using UnicodePlots: Plot, lineplot!, hline!, histogram
using AdaptiveGorilla: count_collisions
import AdaptiveGorilla as AG

using Random
#Random.seed!()

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
    default = :PERF

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

# World model parameters; See "?InertiaWM" for documentation.
WM = load_wm_from_toml("$(@__DIR__)/params/wm.toml")

################################################################################
# General Experiment Parameters
################################################################################

# which dataset to run
DATASET = "study2"
DPATH   = "/spaths/datasets/$(DATASET)/dataset.json"
SCENE   = PARAMS["scene"]
FRAMES  = 240

LONE_PARENT = true
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

function run_model!(pbar, exp, render=false)
    # Initializes the agent
    # (Done from scratch each time to avoid bugs / memory leaks)
    agent = load_agent(MODEL_PARAMS, exp.init_query)
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
        render && render_agent_state(exp, agent, t, out)
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
        CHAINS * (FRAMES-1);
        desc="Running $(MODEL) model...", dt = 1.0)
    plot = Plot(;
                title="Color: $(SWAP_COLORS ? :Dark : :Light) | " *
                    "Parent: $(LONE_PARENT ? :Lone : :Group)",
                xlabel = "t",
                ylabel = "Pr(Notice)",
                xlim = (1, FRAMES-1),
                ylim = (0, 1),
                width = 60, height=20,
                )
    hline!(plot,
           NOTICE_P_THRESH,
           name = "Threshold")
    ndetected = Vector{Int64}(undef, CHAINS)
    experiment = TEnsExp(DPATH, WM, SCENE, SWAP_COLORS, LONE_PARENT, FRAMES;
                         show_gorilla = SHOW_GORILLA)
    gt_count = count_collisions(experiment)
    @show gt_count
    collision_counts = Vector{Float64}(undef, CHAINS)

    Threads.@threads for c = 1:CHAINS
    # for c = 1:CHAINS
        results = run_model!(pbar, experiment, c == 1)
        RENDER && show(results; allrows=true)
        # println()
        collision_counts[c] = last(results[!, :collision_p])
        ndetected[c] = count(results[!, :gorilla_p] .> NOTICE_P_THRESH)
        lineplot!(plot,
                  results[!, :frame],
                  results[!, :gorilla_p])
    end
    finish!(pbar)
    display(plot)
    RENDER || display(
        histogram(ndetected, nbins=15,
                  title = "Frames noticed",
                  vertical = true)
    )
    RENDER ?
        println("Collision counts: $(collision_counts)") :
        display(histogram(collision_counts, nbins=5, vertical=true,
                  title = "Collision counts"))
    return nothing
end;

main();
