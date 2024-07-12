# INSTRUCTIONS:
#
# 1) follow the installation instructions in the README
# 2) start Julia (navigate to the repo directory then run `julia --project=.`)
# 3) Implement the requested functions (see below for more details)
# 4) When you are ready to test your code run `include("scripts/visualize.jl")`
#
# You do not need to implement everything before running the script.
# You should check your work by running the script often.
#
# Remember, if you get unexpected behavior, you can restart julia repeat 2-4 above.
# Have fun!

using Gen
using Luxor
using MOTCore
using AdaptiveGorilla
using AdaptiveGorilla: inertia_init
import MOTCore.paint


"""
Initializes a canvas for the world model and state.
"""
function MOTCore.paint(p::InitPainter, wm::InertiaWM, st::InertiaState)
    # TODO: Step 1
    # HINT: see below as an example
    # https://github.com/CNCLgithub/MOTCore.jl/blob/master/src/scholl/scholl.jl#L163-L168
    return nothing
end

"""
Applies the painter to each element in the world state
"""
function MOTCore.paint(p::Painter, st::InertiaState)
    # TODO: Step 2
    # HINT: see below as an example (different than above)
    # https://github.com/CNCLgithub/MOTCore.jl/blob/master/src/scholl/scholl.jl#L170-L175
    return nothing
end


"""
Applies the Object painter to an InertiaSingle
"""
function MOTCore.paint(p::ObjectPainter, obj::InertiaSingle)
    # TODO: Step 3 (optionally also implement for InertiaEnsemble)
    # HINT: see below as an example (different than above)
    # https://github.com/CNCLgithub/MOTCore.jl/blob/master/src/render/painters/painters.jl#L23-L27
    return nothing
end

"""
Applies the `IDPainter` to each thing in the world state
"""
function MOTCore.paint(p::IDPainter, st::InertiaState)
    # TODO: Step 4
    # NOTE: this is a more specific implementation than step 2
    # HINT: see below as an example (different than above)
    # https://github.com/CNCLgithub/MOTCore.jl/blob/master/src/scholl/scholl.jl#L178-L184
    return nothing
end


# START HERE:
# Look through the comments to understand your tasks
# If all goes well you should find a cool image in
# the repo's main directory.
function test_render_frame()
    # Instantiate a world model
    # and sample an initial state
    wm = InertiaWM()
    state = inertia_init(wm)
    # Initialize the painters
    init = InitPainter(path = "test.png",
                       background = "white")
    objp = ObjectPainter()
    idp = IDPainter()
    # Apply the painters
    paint(init, wm, state) # TODO: implement Step 1
    paint(objp, state) # TODO: implement Steps 2 + 3
    paint(idp, state) # TODO: implement Step 4
    finish()
    return nothing
end

test_render_frame();

# Once you have impemented the above functions,
# uncomment the following section and give it a go.
# function test_render_scene()
#     wm = InertiaWM()
#     _, states = wm_inertia(10, wm)
#     render_scene(wm, states, "states")
#     return nothing
# end
# test_render_scene();
