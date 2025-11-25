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
    wm = InertiaWM(object_rate = 8.0,
                   area_width = 720.0,
                   area_height = 480.0,
                   birth_weight = 0.01,
                   single_size = 5.0,
                   single_noise = 1.0,
                   single_cpoisson_log_penalty = -3000.0,
                   stability = 0.70,
                   vel = 4.5,
                   force_low = 10.0,
                   force_high = 20.0,
                   material_noise = 0.01,
                   ensemble_var_shift = 0.05)

    frames = 190
    trial_idx = 4
    # dpath = "/spaths/datasets/most/dataset.json"
    # dpath = "/spaths/datasets/load_curve/dataset.json"
    # exp = MostExp(dpath, wm, trial_idx,
    #               Dark, frames, true; ntarget=4)
    dpath = "/spaths/datasets/target_ensemble/2025-06-09_W96KtK/dataset.json"
    exp = TEnsExp(dpath, wm, trial_idx,
                  false, true, frames)
    pf = AdaptiveParticleFilter(particles = 5)
    hpf = HyperFilter(;dt=18, pf=pf, h=5)
    perception = PerceptionModule(hpf, exp.init_query)
    # attention = AttentionModule(
    #     AdaptiveComputation(;
    #                         itemp=10.0,
    #                         # base_steps=24,
    #                         # load = 0,
    #                         base_steps=8,
    #                         load = 16,
    #                         buffer_size = 2000,
    #                         map_metric=WeightedEuclidean(S3V(0.1, 0.1, 0.8)),
    #                         )
    # )
    # attention = AttentionModule(UniformProtocol(; moves=24))
    attention = AttentionModule(AdaptiveComputation(; base_steps=8, load=16, buffer_size=2000,
                                                    itemp = 8.0, nns = 10))
    planning = PlanningModule(CollisionCounter(; mat=Light, cooldown=8))
    memory = MemoryModule(
        HyperResampling(;
                        perception = perception,
                        fitness = MhoFitness(;att = attention, beta=3.5,
                                             complexity_mass=10.0, complexity_factor = 6.0),
                        kernel = SplitMergeKernel(;heuristic=MhoSplitMerge(;att = attention)),
                        # fitness = MLLFitness(),
                        # kernel = StaticRKernel(),
                        tau = 1.0,
                        ))
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
    gt_count = AdaptiveGorilla.count_collisions(exp)

    # Profile.init(;delay = 0.01)
    # Profile.clear()
    # @profile for t = 1:(frames - 1)
    @time for t = 1:(frames - 1)
        @show t
        agent_step!(agent, t, get_obs(exp, t))
        _results = run_analyses(exp, agent)
        _results[:frame] = t
        push!(results, _results)
        render && render_agent_state(exp, agent, t, out)
    end


    # Dissect internal of perception after last frame
    visp, visstate = mparse(agent.perception)
    for i = 1:visp.h
        chain = visstate.chains[i]
        particles = chain.state
        mll = AdaptiveGorilla.marginal_ll(particles.traces[1])
        mll = zeros(length(mll))
        for x = particles.traces
            mll .+= AdaptiveGorilla.marginal_ll(x)
        end
        mll .*= 1.0 / length(particles.traces)
        @show mll
    end

    @show gt_count
    # results = DataFrames.select(results, [:frame, :gorilla_p, :birth_p])
    show(results; allrows=true)

    return results
end

test_agent();
