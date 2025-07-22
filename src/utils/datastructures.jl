export HashMap

"""
A container to organize 1D samples (e.g., task relevance) in R^3 coordinate space.
"""
mutable struct HashMap{K, V}
    coords::CircularBuffer{K}
    samples::CircularBuffer{V}
    new_coords::CircularBuffer{K}
    new_samples::CircularBuffer{V}
    map::Union{Nothing, KDTree}
end

Base.isempty(x::HashMap) = isnothing(x.map)

function Base.empty!(x::HashMap)
    empty!(x.coords)
    empty!(x.samples)
    empty!(x.new_coords)
    empty!(x.new_samples)
    x.map = nothing
    return x
end

HashMap(K, V, n::Int) = HashMap(CircularBuffer{K}(n),
                                CircularBuffer{V}(n),
                                CircularBuffer{K}(n),
                                CircularBuffer{V}(n),
                                nothing)

function push_sample!(m::HashMap{K, V}, coord::K, sample::V) where {K, V}
    push!(m.new_coords, coord)
    push!(m.new_samples, sample)
    return nothing
end

function fit_map!(m::HashMap, metric)
    if isempty(m.new_coords) || isempty(m.new_samples)
        m.map = nothing
    else
        # tc = m.coords
        # ts = m.samples
        # m.coords = m.new_coords
        # m.samples = m.new_samples
        # m.new_coords = tc
        # m.new_samples = ts
        # copyto!(m.coords, m.new_coords)
        # copyto!(m.samples, m.new_samples)
        # TODO: Implement Base.copyto!
        m.coords = deepcopy(m.new_coords)
        m.samples = deepcopy(m.new_samples)
        m.map = KDTree(m.coords, metric)
    end
    return nothing
end

"""
Determine the value of `coord`, storing intermediate values in `idxs` and `dists`.
"""
function integrate!(idxs::Vector{Int32},
                    dists::Vector{Float32}, 
                    coord::K,
                    sm::HashMap{K, V}) where {K, V}
    k = min(length(idxs), length(dists))
    knn!(idxs, dists, sm.map, coord, k)
    x = -Inf
    @inbounds for j = 1:k
        idx = idxs[j]
        d = max(dists[j], one(V)) # in case d = 0
        x = logsumexp(x, sm.samples[idx] - log(d))
    end
    x - log(k)
    return x
end
