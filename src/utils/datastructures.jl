export SpatialMap

"""
A container to organize 1D samples (e.g., task relevance) in R^3 coordinate space.
"""
mutable struct SpatialMap
    coords::CircularBuffer{S3V}
    samples::CircularBuffer{Float64}
    new_coords::CircularBuffer{S3V}
    new_samples::CircularBuffer{Float64}
    map::Union{Nothing, KDTree}
end

Base.isempty(x::SpatialMap) = isnothing(x.map)

function Base.empty!(x::SpatialMap)
    empty!(x.coords)
    empty!(x.samples)
    empty!(x.new_coords)
    empty!(x.new_samples)
    x.map = nothing
    return x
end

SpatialMap(n::Int) = SpatialMap(CircularBuffer{S3V}(n),
                                CircularBuffer{Float64}(n),
                                CircularBuffer{S3V}(n),
                                CircularBuffer{Float64}(n),
                                nothing)

function push_sample!(m::SpatialMap, coord::S3V, sample::Float64)
    push!(m.new_coords, coord)
    push!(m.new_samples, sample)
    return nothing
end

function fit_map!(m::SpatialMap, metric)
    if isempty(m.new_coords) || isempty(m.new_samples)
        m.map = nothing
    else
        tc = m.coords
        ts = m.samples
        m.coords = m.new_coords
        m.samples = m.new_samples
        m.new_coords = tc
        m.new_samples = ts
        m.map = KDTree(m.coords, metric)
    end
    return nothing
end

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
    x - log(k)
    return x
end
