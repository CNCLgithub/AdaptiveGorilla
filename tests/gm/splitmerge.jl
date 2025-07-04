using Gen
using AdaptiveGorilla
using AdaptiveGorilla: apply_merge, InertiaSingle, InertiaEnsemble
import MOTCore.paint
using Luxor: finish

function test_merges()
    a = InertiaSingle(Light,
                      [0., 0.],
                      [1., 1.],
                      5.0)
    b = InertiaEnsemble(4.0,
                        [0.333, 0.666],
                        [30., 0.],
                        100.,
                        [2.0, 0.0])

    c = apply_merge(a, b)


    objp = ObjectPainter()
    idp = IDPainter()

    init = InitPainter(path = "/spaths/tests/splitmerge-test-1.png",
                       background = "black")
    MOTCore.paint(init, 200.0, 200.0)
    MOTCore.paint(objp, a)
    MOTCore.paint(objp, b)
    MOTCore.paint(objp, c)
    finish()

    display(a)
    display(b)
    display(c)
    return nothing
end

test_merges()
