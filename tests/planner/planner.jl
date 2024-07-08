using MOTCore
using AdaptiveGorilla
using AdaptiveGorilla: birth_single,
    InertiaSingle,
    Light,
    S2V


function test_energy()
    wm = InertiaWM()
    # walls = MOTCore.init_walls(wm.area_width)
    # wall = first(walls)
    wall = Wall(200.0, S2V(-1, 0.0))
    s = InertiaSingle(Light, [-185., 0.0], [-10.0, 0.0], 10.0)
    display(wall)
    display((s.pos, s.vel))
    @show energy(wall, s)
end

test_energy()
