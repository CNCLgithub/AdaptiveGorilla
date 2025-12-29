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
    default = 32

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

MODEL = PARAMS["model"]
MODEL_PARAMS = "$(@__DIR__)/models/$(MODEL).toml"

WM = load_wm_from_toml("$(@__DIR__)/models/wm.toml")
# World model parameters; See "?InertiaWM" for documentation.
# WM = InertiaWM(;
#                object_rate = 8.0,
#                area_width = 720.0,
#                area_height = 480.0,
#                birth_weight = 0.01,
#                single_size = 5.0,
#                single_noise = 0.15,
#                single_cpoisson_log_penalty = 1.1,
#                stability = 0.75,
#                vel = 4.5,
#                force_low = 3.0,
#                force_high = 10.0,
#                material_noise = 0.01,
#                ensemble_var_shift = 0.1)

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
COLORS = [Light, Dark]

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
    pbar = Progress(2 * CHAINS * (FRAMES-1);
                    desc="Running $(MODEL) model...", dt = 2.0)
    for color = COLORS
        experiment = MostExp(DPATH, WM, SCENE, color, FRAMES, SHOW_GORILLA)
        gt_count = count_collisions(experiment)
        Threads.@threads for c = 1:CHAINS
            ndetected, expected_count = run_model!(pbar, experiment)
            count_error = abs(gt_count - expected_count) / gt_count
            push!(result,
                  (scene = SCENE,
                   color = color == Light ? :light : :dark,
                   chian = c,
                   ndetected = ndetected,
                   expected_count = expected_count,
                   count_error = count_error))
        end
    end
    finish!(pbar)
    out_dir = "/spaths/experiments/$(DATASET)/$(MODEL)-$(ANALYSIS)/scenes"
    isdir(out_dir) || mkpath(out_dir)
    df = DataFrame(result)
    CSV.write("$(out_dir)/$(SCENE).csv", DataFrame(result))
    count_f = x -> count(>(24.0), x) / CHAINS
    display(combine(groupby(df, [:scene, :color]), :ndetected => count_f))
    return nothing
end;

main();
