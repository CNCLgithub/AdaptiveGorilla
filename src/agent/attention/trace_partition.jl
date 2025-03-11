export TracePartition,
    latent_size,
    get_coord,
    select_prop,
    InertiaPartition

""" A procedure to selectively process traces """
abstract type TracePartition{T<:Gen.Trace} end

"""
    latent_size(p::TracePartition{T}, tr::T) where {T<:Gen.Trace}

Determine the number of latents exposed for selective processing.
"""
function latent_size end


"""
    get_coord(p::TracePartition{T}, tr::T, i::Int) where {T<:Gen.Trace}

Get the representational coordinate of latent `i` in the trace.
"""
function get_coord end


"""
    select_prop(p::TracePartition{T}, tr::T, i::Int) where {T<:Gen.Trace}

Get a valid proposal function for the `i`th latent in the trace.
"""
function select_prop end

@with_kw struct InertiaPartition <: TracePartition{InertiaTrace}
    single_prop::Function = single_ancestral_proposal
    ensemble_prop::Function = ensemble_ancestral_proposal
    baby_prop::Function = baby_ancestral_proposal
    singles::Bool = true
    ensemble::Bool = true
    baby::Bool = false
end

function latent_size(partition::InertiaPartition, tr::InertiaTrace)
    t = first(get_args(tr))
    state = tr[:kernel => t]
    n = 0
    if partition.singles
        n += length(state.singles)
    end
    if partition.ensemble
        n += length(state.ensembles)
    end
    if partition.baby
        n += 1
    end
    return n
end

function get_obj(partition::InertiaPartition,
        trace::InertiaTrace, i::Int)

    t = first(get_args(trace))
    state = trace[:kernel => t]
    ns = length(state.singles)
    ne = length(state.ensembles)
    idx = i
    if !partition.singles
        idx += ns # jump past singles
    end
    if !partition.ensemble
        idx += ne # jump past ensembles
    end
    x = if idx <= ns
        state.singles[idx]
    elseif isbetween(idx, ns + 1, ne)
        state.ensembles[idx - ns]
    else
        error("Index $(i) not in trace")
    end
    return x
end

get_coord(x::InertiaSingle) = S3V(get_pos(x)..., Float64(Int(x.mat)))
get_coord(x::InertiaEnsemble) = S3V(get_pos(x)..., mean(x.matws))

function get_coord(partition::InertiaPartition,
                   trace::InertiaTrace, i::Int)
    t = first(get_args(trace))
    state = trace[:kernel => t]
    ns = length(state.singles)
    ne = length(state.ensembles)
    idx = i
    if !partition.singles
        idx += ns # jump past singles
    end
    if !partition.ensemble
        idx += ne # jump past ensembles
    end
    coord = if idx <= ns
        x = state.singles[idx]
        get_coord(x)
    elseif isbetween(idx, ns + 1, ne)
        y = state.ensembles[idx - ns]
        get_coord(y)
    else
        S3V(0., 0., 0.) # REVIEW: not sure where baby should go
    end
end

function select_prop(partition::InertiaPartition,
                     trace::InertiaTrace, i::Int)
    t = first(get_args(trace))
    state = trace[:kernel => t]
    ns = length(state.singles)
    ne = length(state.ensembles)
    idx = i
    if !partition.singles
        idx += ns # jump past singles
    end
    if !partition.ensemble
        idx += ne # jump past ensembles
    end
    prop = if idx <= ns
        tr -> single_ancestral_proposal(tr, idx)
        # bernoulli(0.5) ?
        #     tr -> single_ancestral_proposal(tr, idx) :
        #     tr -> baby_local_transform(tr, idx)
    elseif isbetween(idx, ns + 1, ne)
        tr -> ensemble_ancestral_proposal(tr, idx - ns)
    else
        baby_ancestral_proposal
    end
end
