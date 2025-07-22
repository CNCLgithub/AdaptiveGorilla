

function _interpolate_color(a::S3V, b::S3V, w::Float64)
    a + (w .* (b - a))
end

using Luxor: sethue, Point, circle, setopacity

function render_attention(att::MentalModule{AdaptiveComputation})
    protocol, state = mparse(att)
    isempty(state.dPi) && return nothing
    npoints = length(state.dPi.samples)
    ws = softmax(collect(state.dPi.samples), protocol.itemp)
    lmul!(1.0 / maximum(ws), ws)
    for i = 1:npoints
        coord = state.dPi.coords[i]
        x,y,c = coord
        sample = ws[i]
        r,g,b = _interpolate_color(S3V(0., 0., 1.), S3V(1., 0., 0.), sample)
        # Luxor commands
        setopacity(0.1)
        sethue(r,g,b)
        point = Point(x, -y)
        circle(point, 5.0, :fill)
    end
    return nothing
end
