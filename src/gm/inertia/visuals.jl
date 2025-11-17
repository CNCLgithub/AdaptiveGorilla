using Gen
using MOTCore
using Parameters
using MOTCore: _draw_circle, _draw_text
using AdaptiveGorilla
using AdaptiveGorilla: inertia_init, InertiaSingle, InertiaEnsemble
using Luxor: origin, background, Drawing, sethue, box, Point
import MOTCore.paint


"""
Initializes a canvas for the world model and state.
"""
function MOTCore.paint(p::InitPainter, wm::InertiaWM)
    @unpack area_width, area_height, display_border = wm
    Drawing(area_width, area_height, p.path)
    origin()
    background(p.background)
    # sethue(0.2, 0.2, 0.2)
    # box(Point(0, 0),
    #     area_width - 2 * display_border,
    #     area_height - 2 * display_border,
    #     action = :stroke)
end

function MOTCore.paint(p::InitPainter, area_width, area_height)
    Drawing(area_width, area_height, p.path)
    origin()
    background(p.background)
end

function MOTCore.paint(p::ObjectPainter, obj::InertiaEnsemble)

    # TODO: figure out how to visualize this (dash around emsemble, see its center( a dot?),
    # its cardinality (colour change or a number) )

    # println("\nobj_var $(obj.var)")
    std = sqrt(obj.var)
    w = round(obj.matws[1]; digits = 2)
    w = clamp(w, .2, .8)
    color = (w, w, w)
    _draw_circle(get_pos(obj), 1.0 * std, color;
                 style = :stroke)
    _draw_circle(get_pos(obj), 4.0 * std, color;
                 style = :stroke)
    rte = round(obj.rate; digits = 2)
    _draw_text("λ $(rte)", get_pos(obj))
    return nothing
end

function MOTCore.paint(p::Painter, tr::InertiaTrace,
                       ws::Vector{Float64} = Float64[])
    _, wm, _ = get_args(tr)
    state = get_last_state(tr)
    paint(p, wm, state, ws)
    return nothing
end

"""
Applies the painter to each element in the world state
"""
function MOTCore.paint(p::Painter, wm::InertiaWM, st::InertiaState,
                       ws::Vector{Float64} = Float64[])
    for e = st.ensembles
        MOTCore.paint(p, e)
    end

    for o in st.singles
        MOTCore.paint(p, o)
    end

    # visualize birth
    obj_counts = object_count(st)
    # if obj_counts > wm.object_rate && !isempty(st.singles)
    #     pos = get_pos(st.singles[end])
    #     _draw_circle(pos, 30.0, "purple", opacity = 0.4)
    # end

    # visualize attention
    ns = length(st.singles)
    ne = length(st.ensembles)
    if length(ws) == ns + ne
        @inbounds for i = 1:ns
            pos = get_pos(st.singles[i])
            radius = 40.0 * ws[i]
            _draw_circle(pos, radius, "red", opacity = 0.4)
        end

        @inbounds for i = 1:ne
            pos = get_pos(st.ensembles[i])
            radius = 40.0 * ws[i + ns]
            _draw_circle(pos, radius, "red", opacity = 0.4)
        end
    end
    return nothing
end


function MOTCore.paint(state::InertiaState, ws::Vector{Float64})
    @inbounds for i = eachindex(state.singles)
        pos = get_pos(state.singles[i])
        radius = 40.0 * ws[i]
        _draw_circle(pos, radius, "red", opacity = 0.8)
    end

    # pos = get_pos(state.ensemble)
    # radius = 40.0 * ws[end]
    # _draw_circle(pos, radius, "red", opacity = 0.8)
    return nothing
end


"""
Applies the Object painter to an InertiaSingle
"""
function MOTCore.paint(p::ObjectPainter, obj::InertiaSingle)
    color = obj.mat == Dark ? (0.1, 0.1, 0.1) : (0.9, 0.9, 0.9)
    _draw_circle(get_pos(obj), obj.size, color,
                 opacity = p.alpha; style = :stroke)
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
        MOTCore.paint(p, st.singles[i], i)
    end
    return nothing
end
