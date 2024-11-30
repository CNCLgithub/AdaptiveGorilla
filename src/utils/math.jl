export softmax,
    softmax!

"""
$(TYPEDSIGNATURES)

Computes softmax of an array, with temperature `t`.
"""
function softmax(x::Array{Float64}, t::Float64 = 1.0)
    out = similar(x)
    softmax!(out, x, t)
    return out
end

function softmax!(out::Array{Float64}, x::Array{Float64}, t::Float64 = 1.0)
    nx = length(x)
    maxx = maximum(x)
    sxs = 0.0

    if maxx == -Inf
        out .= 1.0 / nx
        return nothing
    end

    @inbounds for i = 1:nx
        out[i] = @fastmath exp((x[i] - maxx) / t)
        sxs += out[i]
    end
    rmul!(out, 1.0 / sxs)
    return nothing
end

function inbounds(pos::S2V, bbmin::S2V, bbmax::S2V)
    x, y = pos
    bbmin[1] <= x &&
        x <= bbmax[1] &&
        bbmin[2] <= y &&
        y <= bbmax[2]
end


# adapted from
# https://github.com/JeffreySarnoff/AngleBetweenVectors.jl/blob/01c91199fef4d4e8b9178c81662c98ea7f3868d6/src/AngleBetweenVectors.jl#L28
function vec2_angle(a::SVector{2, Float64}, b::SVector{2, Float64})
    a = normalize(a)
    b = normalize(b)
    y = a - b
    x = a + b

    a = 2 * atan(norm(y), norm(x))

    !(signbit(a) || signbit(pi - a)) ? a : (signbit(a) ? 0.0 : pi)
end
