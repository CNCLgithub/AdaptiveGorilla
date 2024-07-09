using Gen
using Gen_Compose
using AdaptiveGorilla

# function test_constraints()

#     wm = InertiaWM(area_height = 1200.0,
#                    area_width = 800.0)
#     dpath = "pilot.json"
#     constraints, obs = query_from_dataset(wm, dpath, 1)

#     tr, w = Gen.generate(wm_inertia, (0, wm), constraints)
#     display(get_choices(tr))
#     @show w
#     @show project(tr, select(:init_state => :singles => 1 => :state))
#     return nothing
# end

# test_constraints();


function test_pf()
    wm = InertiaWM(area_height = 1200.0,
                   area_width = 800.0,
                   birth_weight = 0.0001)
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
