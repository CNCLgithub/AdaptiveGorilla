using MOTCore
using AdaptiveGorilla
using AdaptiveGorilla: S2V, colprob_and_agrad, InertiaSingle, InertiaEnsemble

function test_single()
    wall = Wall(400.0, S2V(-1, 0.0))

    s = InertiaSingle(Light,
                      [-250., 0.],
                      [10.0, 0.0],
                      5.0)

    @show colprob_and_agrad(s, wall)

    return nothing
end

test_single();

function test_ensemble()
    wall = Wall(400.0, S2V(-1, 0.0))

    e = InertiaEnsemble(4.0,
                        [0.25, 0.75],
                        [-250., 0.],
                        40.0,
                        [1.0, 0.0])

    @show colprob_and_agrad(e, wall)

    return nothing
end

test_ensemble();
