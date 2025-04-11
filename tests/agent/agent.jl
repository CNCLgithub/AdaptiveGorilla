using Gen
using MOTCore
using DataFrames
using Gen_Compose
using AdaptiveGorilla
using AdaptiveGorilla: S3V
using Distances: WeightedEuclidean

function test_agent()
    render = true
    wm = InertiaWM(area_width = 1240.0,
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
    dpath = "/spaths/datasets/pilot.json"
    trial_idx = 4
    gorilla_idx = 9
    gorilla_color = Dark
    frames = 41
    exp = Gorillas(dpath, wm, trial_idx, gorilla_idx,
                   gorilla_color, frames)
    query = exp.init_query
    pf = AdaptiveParticleFilter(particles = 5)
    hpf = HyperFilter(;dt=6, pf=pf, h=5)
    perception = PerceptionModule(hpf, query)
    attention = AttentionModule(
        AdaptiveComputation(;
                            itemp=0.1,
                            base_steps=18,
                            load = 0,
                            buffer_size = 200,
                            map_metric=WeightedEuclidean(S3V(0.05, 0.05, 0.9)),
                            )
    )
    planning = PlanningModule(CollisionCounter(; mat=Light))
    memory = MemoryModule(AdaptiveGranularity(; tau=1.0,
                                              shift=false), hpf.h)

    # Cool, =)
    agent = Agent(perception, planning, memory, attention)

    out = "/spaths/tests/agent_state"
    isdir(out) || mkpath(out)

    results = DataFrame(
        :frame => Int64[],
        :gorilla_p => Float64[],
        :collision_p => Float64[]
    )
    gt_exp = AdaptiveGorilla.collision_expectation(exp)
    for t = 1:(frames - 1)
        step_agent!(agent, exp, t)
        _results = run_analyses(exp, agent)
        _results[:frame] = t
        push!(results, _results)
        render && render_agent_state(exp, agent, t, out)
    end

    @show gt_exp
    display(results)

    return nothing
end

test_agent();
