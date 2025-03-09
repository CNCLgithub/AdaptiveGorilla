using Gen
using Luxor
using MOTCore
using MOTCore: _draw_circle
using Gen_Compose
using AdaptiveGorilla

function test_pf()
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
    frames = 10
    exp = Gorillas(dpath, wm, trial_idx, gorilla_idx,
                   gorilla_color, frames)

    # att = UniformProtocol()
    att = AdaptiveProtocol()
    proc = AdaptiveParticleFilter(particles = 10,
                                  attention = att)
    nsteps = length(query)
    logger = MemLogger(nsteps)
    chain = run_chain(proc, query, nsteps, logger)
    out = "/spaths/tests/inference"
    isdir(out) || mkpath(out)
    render_inference(wm, logger, out)
end;

test_pf()
