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
    tr = Gen.simulate(DetectionRFS, (es,))
    display(get_choices(tr))
    # xs = DetectionRFS(es)
    # display(xs)
    @show Gen.get_score(tr)
end

test_rfs()
