using Gen
using AdaptiveGorilla
using AdaptiveGorilla: state_prior,
    birth_single,
    birth_ensemble,
    inertia_init


function test_state_prior()
    wm = InertiaWM()
    trace, w = Gen.generate(state_prior, (wm,))

    @show w
    display(get_choices(trace))
    display(get_score(trace))
end

test_state_prior()

function test_birth_single()
    wm = InertiaWM()
    trace, w = Gen.generate(birth_single, (wm,1.0))
    display(get_choices(trace))
    display(get_score(trace))
    @show w
end

test_birth_single()

function test_birth_ensemble()
    wm = InertiaWM()
    trace, w = Gen.generate(birth_ensemble, (wm, 4.0))
    display(get_choices(trace))
    @show w
    display(get_score(trace))
end

test_birth_ensemble()

function test_inertia_init()
    wm = InertiaWM()
    @time trace, w = Gen.generate(inertia_init, (wm,))
    display(get_choices(trace))
    @show w
    display(get_score(trace))
end

test_inertia_init()
