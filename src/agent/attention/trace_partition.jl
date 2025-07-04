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
    state = get_last_state(tr)
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

function get_coord(o::InertiaSingle)
    x, y = get_pos(o)
    S3V(x, y, Float64(Int(o.mat)))
end
function get_coord(x::InertiaEnsemble)
    m = 0.0
    for (i, matw) = enumerate(x.matws)
        m += matw * Float64(i)
    end
    x,y = get_pos(x)
    S3V(x, y, m)
end

function get_coord(partition::InertiaPartition,
                   trace::InertiaTrace, i::Int)
    state = get_last_state(trace)
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
    elseif isbetween(idx, ns + 1, ns+ne)
        y = state.ensembles[idx - ns]
        get_coord(y)
    else
        S3V(0., 0., 0.) # REVIEW: not sure where baby should go
    end
end

function select_prop(partition::InertiaPartition,
                     trace::InertiaTrace, i::Int)
    state = get_last_state(trace)
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
    elseif isbetween(idx, ns + 1, ns+ne)
        tr -> ensemble_ancestral_proposal(tr, idx - ns)
    else
        baby_ancestral_proposal
    end
end

function dissimilarity(tr::InertiaTrace, metric::PreMetric,
                       a::Int64, b::Int64)
    obj_a = object_from_idx(tr, a)
    coord_a = get_coord(obj_a)
    obj_b = object_from_idx(tr, b)
    coord_b = get_coord(obj_b)
    metric(coord_a, coord_b)
end
