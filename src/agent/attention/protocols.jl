export UniformProtocol

@with_kw struct UniformProtocol <: AttentionProtocol
    """Moves over singles"""
    single_block! = ReweightBlock(single_ancestral_proposal, 20)
    """Moves over ensemble"""
    ensemble_block! = ReweightBlock(ensemble_ancestral_proposal, 3)
    """Moves over babies"""
    baby_block! = ReweightBlock(baby_ancestral_proposal, 3)
end

function AuxState(::UniformProtocol)
    EmptyAuxState()
end

function attend!(chain::APChain, p::UniformProtocol)
    @unpack single_block!, ensemble_block!, baby_block! = p
    @unpack proc, state, auxillary = chain
    np = length(state.traces)
    for i = 1:np
        s = state.traces[i]
        # 1. Single
        k = s[:init_state => :k]
        winner = categorical(Fill(1.0 / k, k))
        single_block!(state, i, winner)

        # 2. Ensemble
        bernoulli(0.5) && ensemble_block!(state, i)

        # 3. Baby
        # bernoulli(0.5) && baby_block!(state, i)
    end

    return nothing
end

export AdaptiveComputation,
    AdaptiveAux,
    SpatialMap

@with_kw struct AdaptiveComputation{T} <: AttentionProtocol
    partition::TracePartition{T} = InertiaPartition()
    planner::Planner{T} = CollisionPlanner(; mat = Light)
    divergence::PreMetric = Euclidean()
    map_metric::PreMetric = WeightedEuclidean(S3V(0.075, 0.075, 0.85))
    base_steps::Int64 = 3
    buffer_size::Int64 = 100
end

mutable struct SpatialMap
    coords::CircularBuffer{S3V}
    trs::CircularBuffer{Float64}
    map::Union{Nothing, KDTree}
end

Base.isempty(x::SpatialMap) = isempty(x.coords) || isempty(trs) || isnothing(map)

function Base.empty!(x::SpatialMap)
    empty!(x.coords)
    empty!(x.trs)
    x.map = nothing
    return x
end

SpatialMap(n::Int) = SpatialMap(CircularBuffer{S3V}(n),
                                CircularBuffer{Float64}(n),
                                nothing)

mutable struct AdaptiveAux <: MentalState{AdaptiveComputation}
    dpi::SpatialMap
    ds::SpatialMap
end
AdaptiveAux(n::Int) = AdaptiveAux(SpatialMap(n), SpatialMap(n))

function update_tr!(aux::AdaptiveAux,
                    partition::TracePartition{T},
                    trace::T,
                    j::Int, delta::Float64) where {T}
    coord = get_coord(partition, trace, j)
    # @show coord
    # @show delta
    push!(aux.new_tr.coords, coord)
    push!(aux.new_tr.trs, delta)
    return nothing
end

function update_tr!(aux::AdaptiveAux, p::AdaptiveComputation)
    tr = aux.tr
    new_tr = aux.new_tr
    new_tr.map = KDTree(new_tr.coords, p.map_metric)
    aux.tr = new_tr
    empty!(tr)
    aux.new_tr = tr
    return nothing
end

# TODO: revisit after `importance`
function load(p::AdaptiveComputation, x::AdaptiveAux)
    (isempty(x.dpi) || isempty(x.ds)) && return 0
    x = logsumexp(tr.trs) - log(length(tr.trs))
    if isnan(x)
        display(tr.trs)
    end
    println("Load : $(x)")
    return 20
end

function task_relevance(
        x::AdaptiveAux,
        partition::TracePartition{T},
        trace::T,
        k::Int = 5) where {T<:Gen.Trace}
    n = latent_size(partition, trace)
    @unpack dPi, dS = x
    # REVIEW: case with empty estimate?
    # No info yet -> -Inf
    isempty(tr) && return Fill(0.0, n)
    tr = zeros(n)
    # Preallocating input results
    idxs, dists = zeros(Int32, k), zeros(Float32, k)
    for i = 1:n
        coord = get_coord(partition, trace, i)
        knn!(idxs, dists, tr.map, coord, k)
        gr = -Inf
        @inbounds for j = 1:k
            idx = idxs[j]
            d = max(dists[j], 1) # in case d = 0
            gr = logsumexp(gr, dPi[idx] + dS[idx] - log(d))
        end
        tr[i] = gr
    end
    return tr
end

function importance(partition::TracePartition{T},
                    trace::T,
                    tr::SpatialMap,
                    k::Int = 5) where {T<:Gen.Trace}
    n = latent_size(partition, trace)
    # No info yet -> uniform weights
    isempty(tr) && return Fill(1.0 / n, n)
    ws = task_relevance(partition, trace, tr, k)
    # TODO: parameterize importance temp
    softmax(ws, 10.0)
end

function attend!(chain::APChain, att::MentalModule{A}) where {A<:AdaptiveComputation}
    p, aux = parse(att)

    @unpack partition, base_steps = p
    @unpack state = chain
    @unpack tr = auxillary

    np = length(state.traces)
    l = load(tr) # load is shared across particles

    for i = 1:np # iterate through each particle
        trace = state.traces[i]
        remaining = l + base_steps
        # Stage 2
        while remaining > 0
            # determine the importance of each latent
            # can change across moves
            w = importance(partition, tr, trace)
            j = categorical(w) # select latent
            prop = select_prop(partition, trace, j)

            # Apply computation, determine dS, dPi
            new_trace, alpha = prop(trace)
            dS = min(alpha, 0.)
            if log(rand()) < alpha # update particle
                trace = new_trace
                state.log_weights[i] += alpha
            end
            # REVIEW: continually updating partition map
            # Does this make AC not invertible?
            # - Actually, an internal TRE is update so this
            # does not change the probablities for the current
            # time step
            update_delta_s!(auxillary, partition, trace, j, dS)
            remaining -= 1
        end

        # baby block
        # REVIEW: remove?
        for _ = 1:base_steps
            new_trace, w = baby_ancestral_proposal(trace)
            if log(rand()) < w
                trace = new_trace
                state.log_weights[i] += w
            end
        end
        state.traces[i] = trace
    end

    # swap internal references
    update_tr!(auxillary, p)

    return nothing
end

