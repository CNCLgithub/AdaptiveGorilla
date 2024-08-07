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
        n += 1
    end
    if partition.baby
        n += 1
    end
    return n
end


function get_coord(partition::InertiaPartition,
                   trace::InertiaTrace, i::Int)
    t = first(get_args(trace))
    state = trace[:kernel => t]
    n = length(state.singles)
    idx = i
    if !partition.singles
        idx += n
    end
    if !partition.ensemble
        idx += 1
    end
    coord = if idx <= n
        x = state.singles[idx]
        S3V(get_pos(x)..., Float64(Int(x.mat)))
    elseif idx == n + 1
        y = state.ensemble
        S3V(get_pos(y)..., mean(y.matws))
    else
        S3V(0., 0., 0.) # REVIEW: not sure where baby should go
    end
end

function select_prop(partition::InertiaPartition,
                     trace::InertiaTrace, i::Int)
    t = first(get_args(trace))
    state = trace[:kernel => t]
    n = length(state.singles)
    idx = i
    if !partition.singles
        idx += n
    end
    if !partition.ensemble
        idx += 1
    end
    prop = if idx <= n
        bernoulli(0.5) ?
            tr -> single_ancestral_proposal(tr, idx) :
            tr -> apply_random_walk(tr, baby_local_proposal, (idx,))
    elseif idx == n + 1
        ensemble_ancestral_proposal
    else
        baby_ancestral_proposal
    end
end
