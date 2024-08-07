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

function apply_protocol!(chain::APChain, p::UniformProtocol)
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

export AdaptiveProtocol,
    AdaptiveAux,
    TREstimate

@with_kw struct AdaptiveProtocol{T} <: AttentionProtocol
    partition::TracePartition{T} = InertiaPartition()
    planner::Planner{T} = CollisionPlanner(; mat = Light)
    divergence::PreMetric = Euclidean()
    map_metric::PreMetric = WeightedEuclidean(S3V(0.1, 0.1, 0.8))
    base_steps::Int64 = 3
    buffer_size::Int64 = 100
end

mutable struct TREstimate
    coords::CircularBuffer{S3V}
    trs::CircularBuffer{Float64}
    map::Union{Nothing, KDTree}
end

Base.isempty(x::TREstimate) = isempty(x.coords)

function Base.empty!(x::TREstimate)
    empty!(x.coords)
    empty!(x.trs)
    x.map = nothing
    return x
end

TREstimate(n::Int) = TREstimate(CircularBuffer{S3V}(n),
                                CircularBuffer{Float64}(n),
                                nothing)

mutable struct AdaptiveAux <: AuxillaryState
    tr::TREstimate
    new_tr::TREstimate
end
AdaptiveAux(n::Int) = AdaptiveAux(TREstimate(n), TREstimate(n))

AuxState(p::AdaptiveProtocol) = AdaptiveAux(p.buffer_size)

function update_tr!(aux::AdaptiveAux,
                    partition::TracePartition{T},
                    trace::T,
                    j::Int, nabla::Float64) where {T}
    coord = get_coord(partition, trace, j)
    # @show coord
    # @show nabla
    push!(aux.new_tr.coords, coord)
    push!(aux.new_tr.trs, nabla)
    return nothing
end

function update_tr!(aux::AdaptiveAux, p::AdaptiveProtocol)
    tr = aux.tr
    new_tr = aux.new_tr
    new_tr.map = KDTree(new_tr.coords, p.map_metric)
    aux.tr = new_tr
    empty!(tr)
    aux.new_tr = tr
    return nothing
end


function apply_protocol!(chain::APChain, p::AdaptiveProtocol)
    @unpack partition, planner, divergence, base_steps = p
    @unpack state, auxillary = chain
    @unpack tr = auxillary
    np = length(state.traces)
    l = load(tr)
    for i = 1:np
        trace = state.traces[i]
        w = importance(partition, trace, tr)
        for j = eachindex(w)
            steps = @inbounds ceil(Int32, w[j] * l) + base_steps
            prop = select_prop(partition, trace, j)
            o = plan(planner, trace)
            nabla = -Inf
            # @show j
            for k = 1:steps
                new_trace, dS = prop(trace)
                dS = min(dS, 0.)
                new_o = plan(planner, new_trace)
                dPi = max(-100., log(divergence(o, new_o)))
                dPi = log(divergence(o, new_o))
                nabla = logsumexp(nabla, dPi + dS)
                if isnan(nabla)
                    @show (o, new_o)
                    @show (dPi, dS)
                end
                if log(rand()) < dS
                    trace = new_trace
                    o = new_o
                    state.log_weights[i] += dS
                end
            end
            nabla -= log(steps)
            # @show nabla
            update_tr!(auxillary, partition, trace, j, nabla)
        end
        # baby block
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

function load(tr::TREstimate)
    isempty(tr) && return 0
    x = logsumexp(tr.trs) - log(length(tr.trs))
    if isnan(x)
        display(tr.trs)
    end
    println("Load : $(x)")
    return 20
end

function importance(partition::TracePartition{T}, trace::T, tr::TREstimate,
                    k::Int = 5) where {T<:Gen.Trace}
    n = latent_size(partition, trace)
    # No info yet -> uniform weights
    isempty(tr) && return Fill(1.0 / n, n)
    ws = Vector{Float64}(undef, n)
    # Preallocating input results
    idxs, dists = zeros(Int32, k), zeros(Float32, k)
    for i = 1:n
        coord = get_coord(partition, trace, i)
        knn!(idxs, dists, tr.map, coord, k)
        gr = -Inf
        @inbounds for j = 1:k
            idx = idxs[j]
            # @show idx
            d = max(dists[j], 1) # in case d = 0
            gr = logsumexp(gr, tr.trs[idx] - log(d))
        end
        ws[i] = gr
    end
    importance = softmax(ws, 2.0)
end

function baby_attention_proposal()
end
