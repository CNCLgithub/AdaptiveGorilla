using Gen
using MOTCore
using AdaptiveGorilla
using AdaptiveGorilla: inertia_init, apply_granularity_move,
    InertiaSingle, InertiaEnsemble, birth_ensemble, give_birth,
    MergeMove

function test_merge_single_single()

    wm = InertiaWM()
    cm = choicemap()
    cm[:n] = 2
    trace, _ = generate(inertia_init, (wm,), cm)
    state = get_retval(trace)
    result = apply_granularity_move(MergeMove(1, 2), wm, state)
    a,b = state.singles
    c = result.ensembles[1]
    display(a.pos)
    display(b.pos)
    display(c)

    return nothing
end;


function test_merge_single_ensemble()

    wm = InertiaWM()
    # a = give_birth(wm)
    # b = birth_ensemble(wm, 3.0)
    a = InertiaSingle(Light,
                      [0., 0.],
                      [1., 1.],
                      10.0)
    b = InertiaEnsemble(3.0,
                        [0.99, 0.01],
                        [10., 0.],
                        20.,
                        [2.0, 0.0])


    state = InertiaState([a], [b])
    result = apply_granularity_move(MergeMove(1, 2), wm, state)
    @show length(result.ensembles)
    c = result.ensembles[1]
    display(a)
    display(b)
    display(c)

    return nothing
end;

function test_merge_ensemble_ensemble()

    wm = InertiaWM()
    a = birth_ensemble(wm, 3.0)
    b = birth_ensemble(wm, 3.0)
    state = InertiaState(InertiaSingle[], [a, b])
    # a = InertiaEnsemble(3.0,
    #                     [0.99, 0.01],
    #                     [0., 0.],
    #                     20.,
    #                     [2.0, 0.0])
    # b = InertiaEnsemble(3.0,
    #                     [0.99, 0.01],
    #                     [300., 0.],
    #                     20.,
    #                     [2.0, 0.0])

    result = apply_granularity_move(MergeMove(1, 2), wm, state)
    @show length(result.ensembles)
    c = result.ensembles[1]
    display(a)
    display(b)
    display(c)
    return nothing
end;

test_merge_single_single();
test_merge_single_ensemble();
test_merge_ensemble_ensemble();
