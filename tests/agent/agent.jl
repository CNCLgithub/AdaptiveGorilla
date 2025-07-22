using Gen
using MOTCore
using DataFrames
using Gen_Compose
using AdaptiveGorilla
using AdaptiveGorilla: S3V, retrieve_map, birth_death_transform
using Distances: WeightedEuclidean

# using Profile
# using PProf

function test_agent()
    render = true
    wm = InertiaWM(area_width = 720.0,
                   area_height = 480.0,
                   birth_weight = 0.1,
                   single_size = 5.0,
                   single_noise = 0.25,
                   single_rfs_logweight = -2500.0,
                   stability = 0.65,
                   # vel = 4.0,
                   vel = 4.5,
                   force_low = 1.0,
                   force_high = 3.5,
                   material_noise = 0.01,
                   ensemble_var_shift = 0.05)
    # dpath = "/spaths/datasets/most/dataset.json"
    dpath = "/spaths/datasets/target_ensemble/2025-06-09_W96KtK/dataset.json"
    trial_idx = 1
    frames = 200
    # exp = MostExp(dpath, wm, trial_idx,
    #               Dark, frames)
    exp = TEnsExp(dpath, wm, trial_idx,
                  false, true, frames)
    query = exp.init_query
    pf = AdaptiveParticleFilter(particles = 5)
    hpf = HyperFilter(;dt=18, pf=pf, h=5)
    perception = PerceptionModule(hpf, query)
    attention = AttentionModule(
        AdaptiveComputation(;
                            itemp=10.0,
                            base_steps=10,
                            load = 20,
                            buffer_size = 2000,
                            map_metric=WeightedEuclidean(S3V(0.25, 0.25, 0.5)),
                            )
    )
    planning = PlanningModule(CollisionCounter(; mat=Light))
    adaptive_g = AdaptiveGranularity(; tau=0.5,
                                     shift=true,
                                     size_cost = 1.0)
    memory = MemoryModule(adaptive_g, hpf.h)

    # Cool, =)
    agent = Agent(perception, planning, memory, attention)

    out = "/spaths/tests/agent_state"
    isdir(out) || mkpath(out)

    results = DataFrame(
        :frame => Int64[],
        :gorilla_p => Float64[],
        :collision_p => Float64[],
        :birth_p => Float64[],
    )
    gt_exp = AdaptiveGorilla.collision_expectation(exp)

    # Profile.init(;delay = 0.01)
    # Profile.clear()
    # @profile for t = 1:(frames - 1)
    for t = 1:(frames - 1)
        @show t
        _results = step_agent!(agent, exp, t)
        _results[:frame] = t
        push!(results, _results)
        render && render_agent_state(exp, agent, t, out)
    end


    # Dissect internal of perception after last frame
    visp, visstate = mparse(agent.perception)
    for i = 1:visp.h
        chain = visstate.chains[i]
        particles = chain.state
        ws = AdaptiveGorilla.marginal_ll(particles.traces[1])
        ws = zeros(length(ws))
        for x = particles.traces
            ws .+= AdaptiveGorilla.marginal_ll(x)
        end
        ws .*= 1.0 / length(particles.traces)
        @show ws
    end

    @show gt_exp
    show(results; allrows=true)

    return results
end

test_agent();
