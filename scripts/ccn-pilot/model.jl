using Gen
using ArgParse
using Gen_Compose
using DataFrames, CSV
using AdaptiveGorilla
using Statistics: mean, median
using ProgressMeter

using AdaptiveGorilla: S3V
using Distances: WeightedEuclidean

DATASET = "pilot"
DPATH   = "/spaths/datasets/$(DATASET).json"
# MODEL = :full
# MODEL = :fixed
MODEL = :no_ac_no_mg
FRAMES  = 41
RENDER  = false
G_IDX   = 9
CHAINS  = 10
NTRIALS = 10
GORILLACOLORS = [Light, Dark]

WM = InertiaWM(area_width = 1240.0,
               area_height = 840.0,
               birth_weight = 0.01,
               single_noise = 0.5,
               stability = 0.5,
               vel = 3.5,
               force_low = 1.0,
               force_high = 5.0,
               material_noise = 0.001,
               ensemble_shape = 1.2,
               ensemble_scale = 1.0,
               ensemble_var_shift = 5.0)



function run_trial!(pbar, exp)
    query = exp.init_query
    pf = AdaptiveParticleFilter(particles = 5)
    hpf = HyperFilter(;dt=6, pf=pf, h=5)
    perception = PerceptionModule(hpf, query)
    if MODEL == :no_ac_no_mg
        base_steps = 9
        load = 0
    else
        base_steps = 8
        load = 10
    end
    attention = AttentionModule(
        AdaptiveComputation(;
                            itemp=0.1,
                            base_steps=base_steps,
                            load = load,
                            buffer_size=200,
                            map_metric=WeightedEuclidean(S3V(0.05, 0.05, 0.9)),
                            )
    )
    planning = PlanningModule(CollisionCounter(; mat=Light))
    shift = MODEL == :full
    memory = MemoryModule(HyperResampling(; tau=1.0,
                                              shift=shift), hpf.h)

    agent = Agent(perception, planning, memory, attention)

    ndetected = 0
    colp = 0.0

    for t = 1:(FRAMES - 1)
        _results = step_agent!(agent, exp, t)
        colp = _results[:collision_p]
        if _results[:gorilla_p] > 0.5
            ndetected += 1
        end
        next!(pbar)
    end

    (ndetected, colp)
end

function run_chains!(result::DataFrame, exp::Gorillas, gtcol::Float64)
    return nothing
end

function main()

    result = DataFrame(:trial => Int64[],
                       :color => Int64[],
                       :chain => Int64[],
                       :ndetected => Int64[],
                       :col_error => Float64[])

    pbar = Progress(NTRIALS * 2 * CHAINS * (FRAMES-1);
                    desc="Running $(MODEL) model...")
    for trial_idx = 1:NTRIALS
        for gorilla_color = GORILLACOLORS
            exp = Gorillas(DPATH, WM, trial_idx, G_IDX,
                           gorilla_color, FRAMES)
            gtcol = AdaptiveGorilla.collision_expectation(exp)
            Threads.@threads for c = 1:CHAINS
                ndetected, colp = run_trial!(pbar, exp)
                colerror = abs(gtcol - colp) / gtcol
                push!(result,
                      (trial_idx,
                       Int(gorilla_color),
                       c,
                       ndetected,
                       colerror))
            end
        end
    end
    finish!(pbar)
    CSV.write("/spaths/experiments/$(DATASET)_$(MODEL).csv",
              result)
    return result
end;


result = main();
