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
