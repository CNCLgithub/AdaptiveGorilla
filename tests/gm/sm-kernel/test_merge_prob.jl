using MOTCore
using AdaptiveGorilla
using AdaptiveGorilla: merge_probability, give_birth,
    InertiaSingle, InertiaEnsemble, birth_ensemble

function test_single_single()

    wm = InertiaWM()
    a = give_birth(wm)
    b = give_birth(wm)
    # a = InertiaSingle(Light,
    #                   [0., 0.],
    #                   [1., 1.],
    #                   10.0)
    # b = InertiaSingle(Light,
    #                   [0., 0.],
    #                   [1., 1.],
    #                   10.0)


    @show merge_probability(a, b)

    return nothing
end;


function test_single_ensemble()

    wm = InertiaWM()
    # a = give_birth(wm)
    b = birth_ensemble(wm, 3.0)
    a = InertiaSingle(Light,
                      [0., 0.],
                      [1., 1.],
                      10.0)
    # b = InertiaEnsemble(3.0,
    #                     [0.99, 0.01],
    #                     [10., 0.],
    #                     20.,
    #                     [2.0, 0.0])


    @show merge_probability(a, b)
    @show merge_probability(b, a)

    return nothing
end;

function test_ensemble_ensemble()

    wm = InertiaWM()
    a = birth_ensemble(wm, 3.0)
    b = birth_ensemble(wm, 3.0)
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


    @show merge_probability(a, b)

    return nothing
end;

test_single_single();
test_single_ensemble();
test_ensemble_ensemble();
