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

using AdaptiveGorilla: S3V
using Distances: WeightedEuclidean

################################################################################
# Command Line Interface
################################################################################


s = ArgParseSettings()

@add_arg_table! s begin

    "--restart", "-r"
    help = "Whether to resume inference"
    action = :store_true


    "--nchains", "-n"
    help = "The number of chains to run"
    arg_type = Int
    default = 10

    "model"
    help = "Model Variant"
    arg_type = Symbol
    range_tester = in([:full, :just_ac, :no_ac_no_mg])
    default = :just_ac

    "scene"
    help = "Which scene to run"
    arg_type = Int64
    default = 1

end

PARAMS = parse_args(ARGS, s)

MODEL = PARAMS["model"]

################################################################################
# Model Parameters
################################################################################

# World model parameters; See "?InertiaWM" for documentation.
WM = InertiaWM(area_width = 720.0,
               area_height = 480.0,
               birth_weight = 0.01,
               single_size = 5.0,
               single_noise = 0.15,
               single_rfs_logweight = 1.1,
               stability = 0.75,
               vel = 4.5,
               force_low = 3.0,
               force_high = 10.0,
               material_noise = 0.01,
               ensemble_var_shift = 0.05)


# Perception Hyper particle-filter; See "?HyperFilter"
VIS_HYPER_COUNT = 5
VIS_PARTICLE_COUNT = 5
VIS_HYPER_WINDOW = 12

# Granularity Optimizer; See "?AdaptiveGranularity"
GO_TAU = 1.0
GO_COST = 1.0
GO_SHIFT = MODEL == :full
GO_PROTOCOL =
    AdaptiveGranularity(;
                        tau=GO_TAU,
                        shift=GO_SHIFT,
                        size_cost=GO_COST)

# Adaptive Computation; See "?AdaptiveComputation"

# Distribution of resources.
# In the case of the fixed resource and granularity model, `BASE_STEPS` equates
# for the number of resources used in the adaptive computation variants, where
# LOAD is split across representations
if MODEL == :no_ac_no_mg
    BASE_STEPS = 12
    LOAD = 0
else
    BASE_STEPS = 16
    LOAD = 24
end

AC_TAU = 10.0
AC_MAP_SIZE = 2000
AC_MAP_METRIC = WeightedEuclidean(S3V(0.1, 0.1, 0.8))
AC_PROTOCOL =
    AdaptiveComputation(;
                        itemp=AC_TAU,
                        base_steps=BASE_STEPS,
                        load = LOAD,
                        buffer_size=AC_MAP_SIZE,
                        map_metric=AC_MAP_METRIC,
                        )


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
SWAP_COLORS = [false, true]

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
NOTICE_P_THRESH = 0.15
# The minimum number of frames where Pr(detect gorilla) > NOTICE_P_THRESH in
# order to consider the gorilla detected for that model run.
NOTICE_MIN_FRAMES = 5

################################################################################
# Methods
################################################################################

# Initializes the agent
# (Done from scratch each time to avoid bugs / memory leaks)
function init_agent(query)
    pf = AdaptiveParticleFilter(particles = VIS_PARTICLE_COUNT)
    hpf = HyperFilter(;dt=VIS_HYPER_COUNT, pf=pf, h=VIS_HYPER_COUNT)
    perception = PerceptionModule(hpf, query)
    attention = AttentionModule(AC_PROTOCOL)
    # Count the number of times light objects bounce
    task_objective = CollisionCounter(; mat=Light)
    planning = PlanningModule(task_objective)
    memory = MemoryModule(GO_PROTOCOL, VIS_HYPER_COUNT)

    Agent(perception, planning, memory, attention)
end

function run_model!(pbar, exp)
    agent = init_agent(exp.init_query)
    colp = 0.0
    noticed = 0
    for t = 1:(FRAMES - 1)
        _results = step_agent!(agent, exp, t)
        _results[:frame] = t
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
    result = Tuple[]
    pbar = Progress(
        length(SWAP_COLORS) * length(LONE_PARENT) * CHAINS * (FRAMES-1);
        desc="Running $(MODEL) model...", dt = 1.0)
    for swap = SWAP_COLORS, lone = LONE_PARENT
        exp = TEnsExp(DPATH, WM, SCENE, swap, lone, FRAMES)
        gtcol = AdaptiveGorilla.collision_expectation(exp)
        Threads.@threads for c = 1:CHAINS
            ndetected, colp = run_model!(pbar, exp)
            colerror = abs(gtcol - colp) / gtcol
            push!(result,
                  (SCENE,
                   swap ? :dark : :light,
                   lone ? :lone : :grouped,
                   c,
                   ndetected,
                   colerror))
        end
    end
    finish!(pbar)
    out_dir = "/spaths/experiments/$(DATASET)/$(MODEL)/scenes"
    isdir(out_dir) || mkpath(out_dir)
    df = DataFrame(result, [:scene, :color, :parent, :chain, :ndetected, :error])
    mean = x -> sum(x) / length(x)
    display(combine(groupby(df, [:scene, :color, :parent]), :ndetected => mean))
    CSV.write("$(out_dir)/$(SCENE).csv", df)
    return nothing
end;

main();
