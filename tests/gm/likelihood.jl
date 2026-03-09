using Gen
using AdaptiveGorilla
import AdaptiveGorilla as AG

function ensemble()
    a = AG.InertiaSingle(AG.Dark, AG.S2V(-50.0, 0.0), AG.S2V(-3.0, 0.0), 0.0)
    b = AG.InertiaSingle(AG.Dark, AG.S2V(50.0, 0.0), AG.S2V(3.0, 0.0), 0.0)
    AG.apply_merge(a, b)
end

function test_rfs()
    e = ensemble()
    display(e)
    mix_args = (AG.get_pos(e),            # Avg. 2D position
                AG.get_var(e),            # Ensemble spread
                first(AG.materials(e)),        # Proportion light
                0.01) # variance for color
    ppp = AG.CPoissonElement{AG.Detection}(AG.rate(e), AG.detect_mixture, mix_args, -3000.0)

    xs = choicemap(
        1 => AG.Detection(50.0, 0.0, 2.0),
        2 => AG.Detection(-50.0, 0.0, 2.0)
    )
    w, _ = Gen.assess(AG.DetectionRFS, ([ppp],), xs)
    @show w


    xs = choicemap(
        1 => AG.Detection(50.0, 0.0, 2.0),
        2 => AG.Detection(100.0, 0.0, 2.0)
    )

    w, _ = Gen.assess(AG.DetectionRFS, ([ppp],), xs)
    @show w
end

test_rfs()
