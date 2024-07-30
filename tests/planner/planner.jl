using MOTCore
using AdaptiveGorilla
using AdaptiveGorilla: birth_single,
    InertiaSingle,
    Light,
    S2V,
    inertia_init,
    init_walls


function test_energy()
    wm = InertiaWM()
    # walls = MOTCore.init_walls(wm.area_width)
    # wall = first(walls)
    wall = Wall(400.0, S2V(-1, 0.0))

    s = InertiaSingle(Light,
                      [-347.8895958702888, -388.41042830191117],
                      [10.0, 0.0], 20.0)

    @show energy(wall, s)
    # s = InertiaSingle(Light, [-180., 0.0], [10.0, 0.0], 10.0)
    # @show x1 = energy(wall, s)
    # s = InertiaSingle(Light, [-170., 0.0], [10.0, 0.0], 10.0)
    # @show x2 = energy(wall, s)
    # s = InertiaSingle(Light, [-160., 0.0], [10.0, 0.0], 10.0)
    # @show x3 = energy(wall, s)
    # @show x2 - x1
    # @show x3 - x2

    # s = InertiaSingle(Light, [-100., 0.0], [10.0, 0.0], 10.0)
    # @show energy(wall, s)
    return nothing
end

test_energy();

function test_planner()
    wm = InertiaWM(; area_width = 1240.0,
                   area_height = 840.0)

    walls = init_walls(wm.area_width, wm.area_height)
    s = InertiaSingle(Light,
                      [-347.8895958702888, -388.41042830191117],
                      [10.0, 0.0], 20.0)


    _e = 0.0
    for k = 1:4
        display(walls[k])
        _e += energy(walls[k], s)
        @show _e
        # e = logsumexp(e, _e)
    end

    # state = inertia_init(wm)
    # planner = CollisionPlanner(;mat = Light)
    # @show plan(planner, state)
    return nothing
end;

test_planner();
