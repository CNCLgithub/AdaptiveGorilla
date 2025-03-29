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
    divergence::PreMetric = Euclidean()
    map_metric::PreMetric = WeightedEuclidean(S3V(0.075, 0.075, 0.85))
    base_steps::Int64 = 3
    buffer_size::Int64 = 100
    "Number of nearest neighbors"
    nns::Int64 = 5
    "Importance softmax temperature"
    itemp::Float64 = 1.0
end

mutable struct AdaptiveAux <: MentalState{AdaptiveComputation}
    "Impact of C_k on planning"
    dPi::SpatialMap
    "Impact of C_k on perception"
    dS::SpatialMap
end
AdaptiveAux(n::Int) = AdaptiveAux(SpatialMap(n), SpatialMap(n))

Base.isempty(x::AdaptiveAux) = isempty(x.dPi) || isempty(dS)


function update_dPi!(att::MentalModule{A},
                     obj::InertiaObject,
                     delta::Float64) where {A<:AdaptiveComputation}
    _, aux = parse(att)
    coord = get_coord(obj)
    push!(aux.dPi.coords, coord)
    push!(aux.dPi.samples, delta)
    return nothing
end

function update_dS!(att::MentalModule{A},
        partition::TracePartition{T},
        trace::T,
        j::Int, delta::Float64) where {A<:AdaptiveComputation, T}
    _, aux = parse(att)
    coord = get_coord(partition, trace, j)
    push!(aux.dS.coords, coord)
    push!(aux.dS.samples, delta)
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
    @unpack dPi, dS = x
    n = latent_size(partition, trace)
    # NOTE: case with empty estimate?
    # No info yet -> -Inf
    isempty(x) && return Fill(-Inf, n)
    tr = Vector{Float64}(undef, n)
    # Preallocating reused arrays
    idxs, dists = zeros(Int32, k), zeros(Float32, k)
    for i = 1:n
        coord = get_coord(partition, trace, i)
        _dpi = integrate!(idxs, dists, coord, dPi)
        _ds  = integrate!(idxs, dists, coord, dS )
        tr[i] = _dpi + _ds - log(k)
    end
    return tr
end

function attend!(chain::APChain, att::MentalModule{A}) where {A<:AdaptiveComputation}
    protocol, aux = parse(att)

    @unpack partition, base_steps, nns, itemp = protocol
    @unpack state = chain

    np = length(state.traces)
    l = load(aux) # load is shared across particles

    for i = 1:np # iterate through each particle
        trace = state.traces[i]
        remaining = l + base_steps
        # Stage 2
        while remaining > 0
            # determine the importance of each latent
            # can change across moves
            deltas = task_relevance(aux, partition, trace, nns)
            importance = softmax(deltas, itemp)
            # select latent and C_k
            j = categorical(importance) 
            prop = select_prop(partition, trace, j)
            # Apply computation, estimate dS
            new_trace, alpha = prop(trace)
            dS = min(alpha, 0.)
            if log(rand()) < alpha # update particle
                trace = new_trace
                state.log_weights[i] += alpha
            end
            # NOTE: continually updating partition map
            update_dS!(att, partition, trace, j, dS)
            remaining -= 1
        end

        # baby block
        # NOTE: Not sure if needed
        for _ = 1:base_steps
            new_trace, w = baby_ancestral_proposal(trace)
            if log(rand()) < w
                trace = new_trace
                state.log_weights[i] += w
            end
        end
        state.traces[i] = trace
    end

    return nothing
end

