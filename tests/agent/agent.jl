using Gen
using MOTCore
using Gen_Compose
using AdaptiveGorilla

function test_agent()
    render = true
    wm = InertiaWM(area_width = 1240.0,
                   area_height = 840.0,
                   birth_weight = 0.1,
                   single_noise = 0.5,
                   stability = 0.75,
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
    frames = 12
    exp = Gorillas(dpath, wm, trial_idx, gorilla_idx,
                   gorilla_color, frames)
    query = exp.init_query
    pf = AdaptiveParticleFilter(particles = 10)
    hpf = HyperFilter(;dt=5, pf=pf, h=5)
    perception = PerceptionModule(hpf, query)
    attention = AttentionModule(AdaptiveComputation(; itemp=3.0))
    planning = PlanningModule(CollisionCounter(; mat=Light))
    memory = MemoryModule(AdaptiveGranularity(; tau=0.01), hpf.h)

    # Cool, =)
    agent = Agent(perception, planning, memory, attention)

    out = "/spaths/tests/agent_state"
    isdir(out) || mkpath(out)

    for t = 1:(frames - 1)
        println("On frame $(t)")
        step_agent!(agent, exp, t)
        render && render_agent_state(exp, agent, t, out)
    end

    return nothing
end

test_agent();
