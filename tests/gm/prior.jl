using Gen
using AdaptiveGorilla
using AdaptiveGorilla: state_prior,
    birth_single,
    birth_ensemble,
    inertia_init


function test_state_prior()
    wm = InertiaWM()

    @show state_prior(wm)

    trace = Gen.simulate(state_prior, (wm,))

    display(get_choices(trace))
end

test_state_prior()

function test_birth_single()
    wm = InertiaWM()
    @show birth_single(wm)
    trace = Gen.simulate(birth_single, (wm,))
    display(get_choices(trace))
end

test_birth_single()

function test_birth_ensemble()
    wm = InertiaWM()
    @show birth_ensemble(wm, 4.0)
    trace = Gen.simulate(birth_ensemble, (wm, 4.0))
    display(get_choices(trace))
end

test_birth_ensemble()

function test_inertia_init()
    wm = InertiaWM()
    @show inertia_init(wm)
    trace = Gen.simulate(inertia_init, (wm,))
    display(get_choices(trace))
end

test_inertia_init()
