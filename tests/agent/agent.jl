using Gen
using MOTCore
using Gen_Compose
using AdaptiveGorilla

function test_agent()
    wm = InertiaWM(area_width = 1240.0,
                   area_height = 840.0,
                   birth_weight = 0.1,
                   single_noise = 1.0,
                   stability = 0.90,
                   vel = 3.0,
                   force_low = 1.0,
                   force_high = 5.0,
                   material_noise = 0.01,
                   ensemble_shape = 1.2,
                   ensemble_scale = 1.0)
    dpath = "/spaths/datasets/pilot.json"
    trial_idx = 1
    gorilla_idx = 9
    gorilla_color = Light
    frames = 20
    exp = Gorillas(dpath, wm, trial_idx, gorilla_idx,
                   gorilla_color, frames)
    query = exp.init_query
    pf = AdaptiveParticleFilter(particles = 10)
    hpf = HyperFilter(;dt=frames // 2, pf=pf)
    perception = PerceptionModule(hpf, query)
    attention = AttentionModule(AdaptiveComputation())
    planning = PlanningModule(CollisionCounter(; mat=Light))
    memory = GranularityModule(AdaptiveGranularity(), hpf.h)

    # burn in for granularity
    for t = 1:10
        println("On frame $(t)")
        obs = get_obs(exp, t)
        perceive!(perception, attention, obs)
        attend!(attention, perception)
        plan!(planning, attention, perception)
        assess_granularity!(memory, attention, perception)
    end

    memp, memstate = mparse(memory)
    display(memstate)

    regranularize!(memory, attention, perception)

    # for t = 11:(frames - 1)
    #     println("On frame $(t)")
    #     obs = get_obs(exp, t)
    #     perceive!(perception, attention, obs)
    #     plan!(planning, attention, perception)
    # end
    return nothing
end

test_agent();
