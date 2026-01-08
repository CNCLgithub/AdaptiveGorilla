using Gen
using MOTCore
using DataFrames
using Gen_Compose
using AdaptiveGorilla
using AdaptiveGorilla: inertia_unfold, inertia_init, predict
using AdaptiveGorilla: S3V, retrieve_map, birth_death_transform
using Distances: WeightedEuclidean
using Luxor: finish


function viz_motion()
    wm = InertiaWM(object_rate = 8.0,
                   area_width = 720.0,
                   area_height = 480.0,
                   birth_weight = 0.0,
                   single_size = 5.0,
                   single_noise = 1.00,
                   single_cpoisson_log_penalty = 1.1,
                   stability = 0.75,
                   vel = 4.5,
                   force_low = 3.0,
                   force_high = 10.0,
                   material_noise = 0.01,
                   ensemble_var_shift = 0.1)

    cm = choicemap(:n => 8)
    for i = 1:8
        cm[:singles => i => :material] = i <= 4 ? 1 : 2
    end
    trace, _ = generate(inertia_init, (wm,), cm)
    istate   = get_retval(trace)
    states   = inertia_unfold(120, istate, wm)
    objp = ObjectPainter()
    for t = 1:120
        init = InitPainter(path = "/spaths/tests/visuals/motion-$(t).png",
                           background = "white")
        # setup
        MOTCore.paint(init, wm)
        MOTCore.paint(objp, wm, states[t])
        finish()
    end

    for t = 1:120
        init = InitPainter(path = "/spaths/tests/visuals/appearance-$(t).png",
                           background = "white")
        # setup
        MOTCore.paint(init, wm)
        rfes = predict(wm, states[t])
        xs = DetectionRFS(rfes)
        MOTCore.paint(objp, xs)
        finish()
    end
end

# viz_motion();

function viz_mgwm()
    wm = InertiaWM(object_rate = 8.0,
                   area_width = 720.0,
                   area_height = 480.0,
                   birth_weight = 0.0,
                   single_size = 5.0,
                   single_noise = 1.00,
                   single_cpoisson_log_penalty = 1.1,
                   stability = 0.75,
                   vel = 4.5,
                   force_low = 3.0,
                   force_high = 10.0,
                   material_noise = 0.01,
                   ensemble_var_shift = 0.1)

    cm = choicemap(:n => 8)
    for i = 1:8
        cm[:singles => i => :material] = i <= 4 ? 1 : 2
    end
    trace, _ = generate(inertia_init, (wm,), cm)
    istate   = get_retval(trace)
    cm = choicemap()
    cm[:s0 => :nsm] = 3
    trace, _ = generate(wm_inertia, (120, wm, istate), cm)
    states   = get_retval(trace)
    objp = ObjectPainter()
    for t = 1:120
        init = InitPainter(path = "/spaths/tests/visuals/mg_motion-$(t).png",
                           background = "white")
        # setup
        MOTCore.paint(init, wm)
        MOTCore.paint(objp, wm, states[t])
        finish()
    end

    for t = 1:120
        init = InitPainter(path = "/spaths/tests/visuals/mg_appearance-$(t).png",
                           background = "white")
        # setup
        MOTCore.paint(init, wm)
        rfes = predict(wm, states[t])
        # display(rfes)
        xs = DetectionRFS(rfes)
        # @show length(xs)
        MOTCore.paint(objp, xs)
        finish()
    end
end

viz_mgwm();
