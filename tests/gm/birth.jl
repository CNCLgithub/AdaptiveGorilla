using Gen
using AdaptiveGorilla
using AdaptiveGorilla: inertia_init,
    birth_process,
    give_birth,
    no_birth,
    birth_or_not


function test_birth()
    wm = InertiaWM()
    state = inertia_init(wm)
    # tr = Gen.simulate(give_birth, (wm,))
    # tr = Gen.simulate(birth_or_not, (1, wm))
    tr, w = Gen.generate(birth_process, (wm, state),
                         choicemap(:pregnant => false))
    display(Gen.get_choices(tr))
    @show w

    singles = get_retval(tr)
    @show length(state.singles)
    @show length(singles)
end

test_birth()
