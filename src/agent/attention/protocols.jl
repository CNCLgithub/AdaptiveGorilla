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
    AttentionModule

@with_kw struct AdaptiveComputation{T} <: AttentionProtocol
    partition::TracePartition{T} = InertiaPartition()
    base_steps::Int64 = 3
    buffer_size::Int64 = 100
    "Distance metric in spatial maps"
    map_metric::PreMetric = WeightedEuclidean(S3V(0.1, 0.1, 0.8))
    "Number of nearest neighbors"
    nns::Int64 = 5
    "Importance softmax temperature"
    itemp::Float64 = 3.0
    "Load - constant for now"
    load::Int64 = 20
end

mutable struct AdaptiveAux <: MentalState{AdaptiveComputation}
    "Impact of C_k on planning"
    dPi::HashMap
    "Impact of C_k on perception"
    dS::HashMap
end

AdaptiveAux(n::Int) = AdaptiveAux(HashMap(S3V, Float64, n),
                                  HashMap(S3V, Float64, n))

function AttentionModule(m::AdaptiveComputation)
    MentalModule(m, AdaptiveAux(m.buffer_size))
end

Base.isempty(x::AdaptiveAux) = isempty(x.dPi) || isempty(x.dS)


function update_dPi!(att::MentalModule{A},
                     obj::InertiaObject,
                     delta::Float64) where {A<:AdaptiveComputation}
    _, aux = mparse(att)
    coord = get_coord(obj)
    push_sample!(aux.dPi, coord, delta)
    return nothing
end

function update_dS!(att::MentalModule{A},
        partition::TracePartition{T},
        trace::T,
        j::Int, delta::Float64) where {A<:AdaptiveComputation, T}
    _, aux = mparse(att)
    coord = get_coord(partition, trace, j)
    push_sample!(aux.dS, coord, delta)
    return nothing
end

function update_task_relevance!(att::MentalModule{A}
                                ) where {A<:AdaptiveComputation}
    attp, attstate = mparse(att)
    fit_map!(attstate.dPi, attp.map_metric)
    fit_map!(attstate.dS , attp.map_metric)
    return nothing
end

# TODO: revisit after `importance`
# Can implement by storing running average
# function load(p::AdaptiveComputation, x::AdaptiveAux)
#     # isempty(x) && return 0
#     # x = logsumexp(tr.trs) - log(length(tr.trs))
#     # if isnan(x)
#     #     display(tr.trs)
#     # end
#     # println("Load : $(x)")
#     return 20
# end

function task_relevance(
        x::AdaptiveAux,
        partition::TracePartition{T},
        trace::T,
        k::Int = 25) where {T<:Gen.Trace}
    @unpack dPi, dS = x
    n = latent_size(partition, trace)
    # NOTE: case with empty estimate?
    # No info yet -> -Inf
    isempty(x) && return fill(-Inf, n)
    tr = Vector{Float64}(undef, n)
    # Preallocating reused arrays
    idxs, dists = zeros(Int32, k), zeros(Float32, k)
    for i = 1:n
        coord = get_coord(partition, trace, i)
        _dpi = integrate!(idxs, dists, coord, dPi)
        # tr[i] = _dpi
        _ds  = integrate!(idxs, dists, coord, dS )
        tr[i] = _dpi + _ds
    end
    return tr
end

function attend!(att::MentalModule{A}, vis::MentalModule{V}
                 ) where {A<:AttentionProtocol, V<:HyperFilter}
    visp, visstate = mparse(vis)
    update_task_relevance!(att)
    for i = 1:visp.h
        chain = visstate.chains[i]
        attend!(chain, att)
    end
    return nothing
end

function attend!(chain::APChain, att::MentalModule{A}) where {A<:AdaptiveComputation}
    protocol, aux = mparse(att)

    @unpack partition, base_steps, nns, itemp, load = protocol
    @unpack state = chain

    np = length(state.traces)
    # l = load(protocol, aux) # load is shared across particles

    for i = 1:np # iterate through each particle
        trace = state.traces[i]
        # determine the importance of each latent
        deltas = task_relevance(aux, partition, trace, nns)
        importance = softmax(deltas, itemp)
        nobj = length(deltas)
        steps_per_obj = round(Int, base_steps / nobj)
        # Stage 2
        # select latent and C_k
        for j = 1:nobj
            steps = steps_per_obj +
                round(Int, load * importance[j])
            for _ = 1:steps
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
            end
        end

        # Localized birth-death
        j = argmax(importance)
        trace = baby_loop!(state, trace, i, j)
        # baby block
        # # TODO: Hyperparameter
        # for _ = 1:base_steps
        #     new_trace, w = baby_ancestral_proposal(trace)
        #     if log(rand()) < w
        #         trace = new_trace
        #         state.log_weights[i] += w
        #     end
        # end
        state.traces[i] = trace
    end

    return nothing
end

function baby_loop!(state, trace, i, j)
    for _ = 1:5
        new_trace, w = bd_loc_transform(trace, j)
        if log(rand()) < w
            trace = new_trace
            state.log_weights[i] += w
            break
        end
    end
    return trace
end
