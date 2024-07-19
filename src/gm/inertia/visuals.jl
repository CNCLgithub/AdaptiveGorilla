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
function MOTCore.paint(p::Painter, st::InertiaState,
                       ws::Vector{Float64} = Float64[])
    paint(p, st.ensemble)

    for o in st.singles
        paint(p, o)
    end

    # visualize attention
    if length(ws) == length(st.singles) + 1
        @inbounds for i = eachindex(st.singles)
            pos = get_pos(st.singles[i])
            radius = 40.0 * ws[i]
            _draw_circle(pos, radius, "red", opacity = 0.8)
        end

        pos = get_pos(st.ensemble)
        radius = 40.0 * ws[end]
        _draw_circle(pos, radius, "red", opacity = 0.8)
    end
    return nothing
end


function MOTCore.paint(state::InertiaState, ws::Vector{Float64})
    display(ws)
    @inbounds for i = eachindex(state.singles)
        pos = get_pos(state.singles[i])
        radius = 40.0 * ws[i]
        _draw_circle(pos, radius, "red", opacity = 0.8)
    end

    pos = get_pos(state.ensemble)
    radius = 40.0 * ws[end]
    _draw_circle(pos, radius, "red", opacity = 0.8)
    return nothing
end


"""
Applies the Object painter to an InertiaSingle
"""
function MOTCore.paint(p::ObjectPainter, obj::InertiaSingle)
    color = obj.mat == Dark ? (0.1, 0.1, 0.1) : (0.9, 0.9, 0.9)
    _draw_circle(get_pos(obj), obj.size, color,
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
    for i in eachindex(st.singles)
        paint(p, st.singles[i], i)
    end
    return nothing
end
