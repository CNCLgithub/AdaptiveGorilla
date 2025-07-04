using Gen
using ArgParse
using Gen_Compose
using ProgressMeter
using DataFrames, CSV
using AdaptiveGorilla
using UnicodePlots

using AdaptiveGorilla: S3V
using Distances: WeightedEuclidean

DATASET = "target_ensemble/2025-06-09_W96KtK"
DPATH   = "/spaths/datasets/$(DATASET)/dataset.json"
MODEL = :full
# MODEL = :fixed
# MODEL = :no_ac_no_mg
FRAMES  = 175
RENDER  = false
CHAINS  = 4
# 24 Conditions total:
# 6 scenes x 2 colors x 2 gorilla parents
NTRIALS = 1
LONE_PARENT = [true, false] # [false, true]
SWAP_COLORS = [false, true] # [false, true]

# World model parameters
WM = InertiaWM(area_width = 720.0,
               area_height = 480.0,
               birth_weight = 0.1,
               single_size = 5.0,
               single_noise = 0.35,
               single_rfs_logweight = -2500.0,
               stability = 0.5,
               vel = 4.5,
               force_low = 1.0,
               force_high = 5.0,
               material_noise = 0.01,
               ensemble_var_shift = 0.1)

# Analysis parameters
NOTICE_THRESH = 0.1

# Initializes the agent
# (Done from scratch each time to avoid bugs)
function init_agent(query)
    pf = AdaptiveParticleFilter(particles = 5)
    hpf = HyperFilter(;dt=18, pf=pf, h=5)
    perception = PerceptionModule(hpf, query)
    if MODEL == :no_ac_no_mg
        base_steps = 9
        load = 0
    else
        base_steps = 8
        load = 20
    end
    map_metric = WeightedEuclidean(S3V(0.05, 0.05, 0.9))
    attention = AttentionModule(
        AdaptiveComputation(;
                            itemp=10.0,
                            base_steps=base_steps,
                            load = load,
                            buffer_size=1000,
                            map_metric=map_metric,
                            )
    )
    # Count the number of times light objects bounce
    task_objective = CollisionCounter(; mat=Light)
    planning = PlanningModule(task_objective)
    shift = MODEL == :full
    granularity_opt =
        AdaptiveGranularity(; tau=1.0, shift=shift, size_cost=40.0)
    memory = MemoryModule(granularity_opt, hpf.h)

    Agent(perception, planning, memory, attention)
end

function run_trial!(pbar, exp)
    agent = init_agent(exp.init_query)
    colp = 0.0
    noticed = 0
    # results = DataFrame(
    #     :frame => Int64[],
    #     :gorilla_p => Float64[],
    #     :collision_p => Float64[],
    #     :birth_p => Float64[],
    # )
    for t = 1:(FRAMES - 1)
        _results = step_agent!(agent, exp, t)

        _results[:frame] = t
        # push!(results, _results)

        colp = _results[:collision_p]
        # @show t
        # display(_results[:gorilla_p])
        if  _results[:gorilla_p] > NOTICE_THRESH
            noticed += 1
        end
        next!(pbar)
    end

    # show(results; allrows=true)
    (noticed, colp)
end

function main()

    result = DataFrame(:trial => Int64[],
                       :color => Symbol[],
                       :parent => Symbol[],
                       :chain => Int64[],
                       :ndetected => Int64[],
                       :col_error => Float64[])
    # result = DataFrame(:trial => Int64[],
    #                    :color => Symbol[],
    #                    :parent => Symbol[],
    #                    :chain => Int64[],
    #                    :notice => Float64[])

    pbar = Progress(
        NTRIALS * length(SWAP_COLORS) *
            length(LONE_PARENT) * CHAINS * (FRAMES-1);
        desc="Running $(MODEL) model...")
    # for trial_idx = 1:NTRIALS, swap = SWAP_COLORS, lone = LONE_PARENT
    trial_idx = 1
    for swap = SWAP_COLORS, lone = LONE_PARENT
        exp = TEnsExp(DPATH, WM, trial_idx, swap, lone, FRAMES)
        gtcol = AdaptiveGorilla.collision_expectation(exp)
        Threads.@threads for c = 1:CHAINS
        # for c = 1:CHAINS
            ndetected, colp = run_trial!(pbar, exp)
            colerror = abs(gtcol - colp) / gtcol
            # @show (trial_idx, lone, c)
            # display(lineplot(notice, xlabel = "time", ylabel = "Pr(Notice)",
            #                  ylim = (0., 1.0),
            #                  title = "Trial: $(trial_idx) | Parent: $(lone) | Chain: $(c)"))
            push!(result,
                  (trial_idx,
                   swap ? :dark : :light,
                   lone ? :lone : :grouped,
                   c,
                   ndetected,
                   colerror))
        end
    end
    finish!(pbar)
    out_dir = "/spaths/experiments/$(DATASET)"
    isdir(out_dir) || mkpath(out_dir)
    CSV.write("$(out_dir)/$(MODEL).csv", result)
    return result
end;


result = main();
