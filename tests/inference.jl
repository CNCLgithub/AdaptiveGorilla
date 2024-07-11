using Gen
using Gen_Compose
using AdaptiveGorilla
using AdaptiveGorilla: baby_ancestral_proposal
using AdaptiveGorilla: birth_process, inertia_init, give_birth

function test_constraints()

    wm = InertiaWM(area_width = 1200.0,
                   area_height = 800.0)
    dpath = "pilot.json"
    constraints, obs = query_from_dataset(wm, dpath, 1)

    tr, w = Gen.generate(wm_inertia, (0, wm), constraints)
    display(get_choices(tr))
    @show w
    @show project(tr, select(:init_state => :singles => 1 => :state))
    return nothing
end

# test_constraints();


function test_birth_prop()
    wm = InertiaWM(area_width = 1200.0,
                   area_height = 800.0,
                   birth_weight = 0.5)
    st = inertia_init(wm)

    # tr = Gen.simulate(birth_process, (wm, st))
    # display(get_choices(tr))
    # selection = AllSelection()
    # w = project(tr, selection)
    # @show w

    tr = Gen.simulate(birth_process, (wm, st))
    display(get_choices(tr))

    # if tr[:pregnant]
    #     selection = select(:pregnant,
    #                        :birth => :baby)
    # else
    #     selection = select(:pregnant)
    # end

    selection = select(:pregnant)
    new_tr, w = regenerate(tr, selection)
    display(get_choices(new_tr))
    @show w

    # tr = simulate(wm_inertia, (1, wm))
    # (new_tr, w) = baby_ancestral_proposal(tr)

    return nothing
end;

test_birth_prop()

function test_pf()
    wm = InertiaWM(area_width = 1200.0,
                   area_height = 800.0,
                   birth_weight = 0.0001,
                   single_noise = 20.0)
    dpath = "pilot.json"
    query = query_from_dataset(wm, dpath, 1, 80)
    att = UniformProtocol()
    proc = AdaptiveParticleFilter(particles = 10,
                                  attention = att)
    nsteps = length(query)
    logger = MemLogger(nsteps)
    chain = run_chain(proc, query, nsteps, logger)
end;

test_pf()
