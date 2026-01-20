################################################################################
# Script to run models on the Most Experiment (Study 1)
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
    default = 1

    "model"
    help = "Model Variant"
    arg_type = Symbol
    range_tester = in(keys(MODEL_VARIANTS))
    default = :mo

    "scene"
    help = "Which scene to run"
    arg_type = Int64
    default = 1

end

PARAMS = parse_args(ARGS, s)

################################################################################
# Model Parameters
################################################################################

# which model variant to run (uncomment 1 of the lines below)
MODEL = PARAMS["model"]
MODEL_PARAMS = "$(@__DIR__)/params/$(MODEL).toml"

WM = load_wm_from_toml("$(@__DIR__)/params/wm.toml")

################################################################################
# ANALYSES
################################################################################

ANALYSIS = PARAMS["analyses"]
SHOW_GORILLA = true # ANALYSIS == :NOTICE

################################################################################
# General Experiment Parameters
################################################################################

# which dataset to run
DATASET = "most"
DPATH   = "/spaths/datasets/$(DATASET)/dataset.json"
SCENE   = PARAMS["scene"]
FRAMES  = 240

# 2 Conditions total: Gorilla Light | Dark
# COLORS = [Light, Dark]
COLORS = [Dark]

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
NOTICE_P_THRESH = 0.50

################################################################################
# Methods
################################################################################

# Initializes the agent
# (Done from scratch each time to avoid bugs / memory leaks)
function init_agent(query)
    load_agent(MODEL_PARAMS, query)
end

function run_model!(pbar, exp)
    agent = load_agent(MODEL_PARAMS, exp.init_query)
    out = "/spaths/tests/most"
    isdir(out) || mkpath(out)

    results = DataFrame(
        :frame => Int64[],
        :gorilla_p => Float64[],
        :collision_p => Float64[],
        :birth_p => Float64[],
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
    pbar = Progress(CHAINS * (FRAMES-1);
                    desc="Running $(MODEL) model...", dt = 1.0)
    for color = COLORS
        experiment = MostExp(DPATH, WM, SCENE, color, FRAMES, SHOW_GORILLA)
        gt_count = count_collisions(experiment)
        @show gt_count
        results = run_model!(pbar, experiment)
        show(results; allrows=true)
        println("\n  ------")
        count_f = x -> count(>(0.25), x) / CHAINS
        @show count_f(results[!, :gorilla_p])
    end
    finish!(pbar)
    return nothing
end;

main();
