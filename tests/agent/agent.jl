using Gen
using MOTCore
using DataFrames
using Gen_Compose
using AdaptiveGorilla
using AdaptiveGorilla: S3V
using Distances: WeightedEuclidean

function test_agent()
    render = true
    wm = InertiaWM(area_width = 720.0,
                   area_height = 480.0,
                   birth_weight = 0.00,
                   single_size = 5.0,
                   single_noise = 0.25,
                   stability = 0.5,
                   vel = 3.0,
                   force_low = 1.0,
                   force_high = 5.0,
                   material_noise = 0.001,
                   ensemble_shape = 1.2,
                   ensemble_scale = 1.0,
                   ensemble_var_shift = 5.0)
    dpath = "/spaths/datasets/target_ensemble/2025-06-09_W96KtK/dataset.json"
    trial_idx = 1
    gorilla_color = Dark
    frames = 200
    # exp = MostExp(dpath, wm, trial_idx,
    #               gorilla_color, frames)
    exp = TEnsExp(dpath, wm, trial_idx,
                  false, false, frames)
    query = exp.init_query
    pf = AdaptiveParticleFilter(particles = 5)
    hpf = HyperFilter(;dt=12, pf=pf, h=5)
    perception = PerceptionModule(hpf, query)
    attention = AttentionModule(
        AdaptiveComputation(;
                            itemp=10.0,
                            base_steps=8,
                            load = 10,
                            buffer_size = 1000,
                            map_metric=WeightedEuclidean(S3V(0.05, 0.05, 0.9)),
                            )
    )
    planning = PlanningModule(CollisionCounter(; mat=Light))
    memory = MemoryModule(AdaptiveGranularity(; tau=1.0,
                                              shift=true), hpf.h)

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
        _results = step_agent!(agent, exp, t)
        _results[:frame] = t
        push!(results, _results)
        render && render_agent_state(exp, agent, t, out)
    end

    @show gt_exp
    show(results; truncate = frames)

    return results
end

test_agent();
