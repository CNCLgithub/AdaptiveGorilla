include("types.jl")
include("math.jl")
include("distributions/distributions.jl")
include("io.jl")

const WALL_ANGLES = [0.0, pi/2, pi, 3/2 * pi]

function init_walls(width::Real, height::Real)
    ws = Vector{Wall}(undef, 4)
    v = S2V(width * 0.5, height * 0.5)
    @inbounds for (i, theta) in enumerate(WALL_ANGLES)
       normal = S2V(cos(theta), sin(theta))
       sign = (-1)^(i > 2)
       ws[i] = Wall(sign * sum(normal .* v), normal)
    end
    return SVector{4, Wall}(ws)
end


import Base.keys

function Base.keys(x::Base.Iterators.Filter)
    Base.keys(collect(x))
end
