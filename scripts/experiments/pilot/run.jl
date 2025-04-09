using Gen
using ArgParse
using Gen_Compose
using DataFrames, CSV
using AdaptiveGorilla
using Statistics: mean

# att_modules = Dict(
#     :ac => AdaptiveComputation,
#     :un => UniformProtocol
# )

# att_configs = Dict(
#     :ac => "$(@__DIR__)/ac.json",
#     :un => "$(@__DIR__)/un.json",
# )

DATASET = "pilot"
DPATH   = "/spaths/datasets/$(DATASET).json"
FRAMES  = 41
RENDER  = false
G_IDX   = 9
CHAINS  = 3
NTRIALS = 1
GORILLACOLORS = [Light, Dark]

WM = InertiaWM(area_width = 1240.0,
               area_height = 840.0,
               birth_weight = 0.1,
               single_noise = 0.5,
               stability = 0.65,
               vel = 3.0,
               force_low = 1.0,
               force_high = 5.0,
               material_noise = 0.001,
               ensemble_shape = 1.2,
               ensemble_scale = 1.0)



function run_trial(exp)
    query = exp.init_query
    pf = AdaptiveParticleFilter(particles = 15)
    hpf = HyperFilter(;dt=10, pf=pf, h=5)
    perception = PerceptionModule(hpf, query)
    attention = AttentionModule(
        AdaptiveComputation(;
                            itemp=3.0,
                            base_steps=5,
                            )
    )
    planning = PlanningModule(CollisionCounter(; mat=Light))
    memory = MemoryModule(AdaptiveGranularity(; tau=1.0), hpf.h)

    agent = Agent(perception, planning, memory, attention)

    ndetected = 0

    for t = 1:(FRAMES - 1)
        println("On frame $(t)")
        step_agent!(agent, exp, t)
        _results = run_analyses(exp, agent)
        @show _results[:collision_p]
        if _results[:gorilla_p] > 0.5
            ndetected += 1
        end
        if ndetected > 10
            break
        end
    end

    return ndetected
end

function main()

    result = DataFrame(:trial => Int64[],
                       :color => Int64[],
                       :chain => Int64[],
                       :ndetected => Int64[])




    for trial_idx = 1:NTRIALS, gorilla_color = GORILLACOLORS
        exp = Gorillas(DPATH, WM, trial_idx, G_IDX,
                       gorilla_color, FRAMES)
        for c = 1:CHAINS
            ndetected = run_trial(exp)
            push!(result, (trial_idx, Int(gorilla_color), c, ndetected))
        end
    end
    display(result)
    grouped = groupby(result, :color)
    display(combine(grouped, :ndetected => mean))
end;


main();
