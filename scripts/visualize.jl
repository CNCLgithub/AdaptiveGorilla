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
using Parameters
using MOTCore: _draw_circle
using AdaptiveGorilla
using AdaptiveGorilla: inertia_init, InertiaSingle, InertiaEnsemble
import MOTCore.paint


"""
Initializes a canvas for the world model and state.
"""
function MOTCore.paint(p::InitPainter, wm::InertiaWM, st::InertiaState) 
    # TODO: Step 1
    @unpack area_width, area_height, display_border = wm
    Drawing(area_width, area_height, p.path)
    Luxor.origin()
    background(p.background)
    sethue(0.2, 0.2, 0.2)
    box(Point(0, 0),
        area_width - 2 * display_border,
        area_height - 2 * display_border,
        action = :stroke)
end

function MOTCore.paint(p::ObjectPainter, obj::InertiaEnsemble)

    # TODO: figure out how to visualize this (dash around emsemble, see its center( a dot?), 
    # its cardinality (colour change or a number) )

    _draw_circle(get_pos(obj), obj.var, "purple", 
         opacity = 1)
        return nothing
    end

"""
Applies the painter to each element in the world state
"""
function MOTCore.paint(p::Painter, st::InertiaState)
    # TODO: Step 2
    paint(p, st.ensemble)

    for o in st.singles
        paint(p, o)
    end
    

    return nothing
end



"""
Applies the Object painter to an InertiaSingle
"""
function MOTCore.paint(p::ObjectPainter, obj::InertiaSingle)
    # TODO: Step 3 (optionally also implement for InertiaEnsemble)
    _draw_circle(get_pos(obj), obj.size, p.dot_color,
         opacity = p.alpha)
        return nothing
    end
    
    #HINT: see below as an example (different than above)
    # https://github.com/CNCLgithub/MOTCore.jl/blob/master/src/render/painters/painters.jl#L23-L27

function MOTCore.paint(id::IDPainter, obj::InertiaSingle, idx::Int64)
    pos = get_pos(obj)
    MOTCore._draw_text("$idx", pos .+ [obj.size, obj.size],
                          size = id.label_size)
    return nothing
end



"""
Applies the `IDPainter` to each thing in the world state
"""
function MOTCore.paint(p::IDPainter, st::InertiaState)
    # TODO: Step 4
    for i in eachindex(st.singles)
        paint(p, st.singles[i], i)
    end
    return nothing
end



    # NOTE: this is a more specific implementation than step 2
    # HINT: see below as an example (different than above)
    # https://github.com/CNCLgithub/MOTCore.jl/blob/master/src/scholl/scholl.jl#L178-L184
    


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
function test_render_scene()
    wm = InertiaWM()
    # display(wm)
    _, states = wm_inertia(120, wm)
    render_scene(wm, states, "states")
    return nothing
end
 test_render_scene();
