using Gen
using AdaptiveGorilla
using AdaptiveGorilla: inertia_init,
    predict,
    DetectionRFS


function test_rfs()
    wm = InertiaWM()
    state = inertia_init(wm)
    es = predict(wm, state)
    display(es)
    display(DetectionRFS(es))
end

test_rfs()
