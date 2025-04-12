using Gen
using MOTCore
using Luxor: finish
using AdaptiveGorilla
using AdaptiveGorilla: inertia_init

function test_render_frame()
    # Instantiate a world model
    # and sample an initial state
    wm = InertiaWM(; area_width = 1240,
                   area_height = 840)
    state = inertia_init(wm)
    # Initialize the painters
    init = InitPainter(path = "/spaths/tests/test.png",
                       background = "white")
    objp = ObjectPainter()
    idp = IDPainter()
    # Apply the painters
    paint(init, wm, state)
    paint(objp, state)
    paint(idp, state)
    finish()
    return nothing
end

test_render_frame();

# Once you have impemented the above functions,
# uncomment the following section and give it a go.
function test_render_scene()
    wm = InertiaWM(; area_width = 1240,
                   area_height = 840)
    # display(wm)
    _, states = wm_inertia(120, wm)
    render_scene(wm, states, "/spaths/tests/states")
    return nothing
end
test_render_scene();
