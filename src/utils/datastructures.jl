export SpatialMap

"""
A container to organize 1D samples (e.g., task relevance) in R^3 coordinate space.
"""
mutable struct SpatialMap
    coords::CircularBuffer{S3V}
    samples::CircularBuffer{Float64}
    map::Union{Nothing, KDTree}
end

Base.isempty(x::SpatialMap) = isempty(x.coords) || isempty(x.samples) || isnothing(x.map)

function Base.empty!(x::SpatialMap)
    empty!(x.coords)
    empty!(x.samples)
    x.map = nothing
    return x
end

SpatialMap(n::Int) = SpatialMap(CircularBuffer{S3V}(n),
                                CircularBuffer{Float64}(n),
                                nothing)
"""
Determine the value of `coord`, storing intermediate values in `idxs` and `dists`.
"""
function integrate!(idxs::Vector{Int32},
                    dists::Vector{Float32}, 
                    coord::S3V,
                    sm::SpatialMap)
    k = min(length(idxs), length(dists))
    knn!(idxs, dists, sm.map, coord, k)
    x = -Inf
    @inbounds for j = 1:k
        idx = idxs[j]
        d = max(dists[j], 1) # in case d = 0
        x = logsumexp(x, sm.samples[idx] - log(d))
    end
    return x
end
