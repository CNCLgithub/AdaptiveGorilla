

abstract type GranularityProtocal end

abstract type GranularityMove end

struct SplitMove <: GranularityMove
    x::Int64
end

struct MergeMove <: GranularityMove
    a::Int64
    b::Int64
end

const SplitMergeMove = Union{SplitMove, MergeMove}

"""
Defines the granularity mapping for a current trace format
"""
@with_kw struct AdaptiveGranularity <: GranularityProtocal
    """Ratio between objectve 1 and 2"""
    m::Float64
    """Number of particles to approx split-merge alpha"""
    alpha_steps::Int64
    """Number of objects to evaluate"""
    object_steps::Int64
end

function granularity_objective(ag::MentalModule{G},
                               ac::MentalModule{A}
                               chain::APChain,
    ) where {G<:AdaptiveGranularity, A<:AdaptiveComputation}
    attp, attx = parse(ac)
    @unpack partition, nns = attp
    @unpack state = chain
    # Running averages
    mag = -Inf
    len = 0.0
    np = length(state)
    @inbounds for i = 1:np
        tr = task_relevance(attx,
                            attp.partition,
                            state.traces[i],
                            attp.nns)
        mag = logsumexp(mag, l2log(tr))
        len += length(tr)
    end
    mag - log(len)
end

function sample_granularity_move(nsingle::Int, nensemble::Int)
    ntotal = nsingle + nensemble
    nsplits = nensemble
    nmerges = ncr(ntotal, 2)
    move = if nensemble > 0 && rand() > 0.5
        # sample which ensemble to split
        SplitMove(rand(1:nensemble))
    else
        # sample which pair to merge
        lex = rand(1:nmerges)
        a, b = combination(ntotal, 2, lex)
        MergeMove(a, b)
    end
end

function regranularize!(mem::MentalModule{M},
                        att::MentalModule{A},
                        vis::MentalModule{V}
    ) where {M<:AdaptiveGranularity,
             A<:AdaptiveComputation,
             V<:HyperFilter}

    hf, vstate = parse(vis)
    ws = zeros(hf.h)
    for i = 1:hf.h
        ws[i] = granularity_objective(ag, vstate.chains[i], astate)
    end

    ws = softmax(ws)
    resample_chains!(vstate, ws)


    for i = 1:hf.h
        ws[i] = granularity_objective(ag, vstate.chains[i], astate)
    end


end

function foo(chain::APChain, p::AdaptiveComputation, g::AdaptiveGranularity)
    # Get object weights
    trs = compute_aligned_task_relevance(chain, p) # TODO

    # greedy approach, look at lowest tr first
    local m::SplitMergeMove, w::Float64
    toconsider = sortperm(trs)[1:g.object_steps]
    # merges
    for i = toconsider[1:(g.object_steps-1)]
        for j = toconsider[2:end]
            _m = Merge(i, j)
            _w = gratio(m, chain, i, trs[i], j, trs[j])
            if w < _w
                m = _m
                w = _w
            end
        end
    end
    # splits
    for i = toconsider
        if cansplit(current_schema, i)
            _m = Split(i)
            _w = gratio(s, chain, i, trs[i])
            if w < _w
                m = _m
                w = _w
            end
        end
    end



    # split-merge
    merge_weights = compute_merge_weights()
    to_merge = resolve_merge(merge_weights, nabla)

    split_weights = compute_split_weights(to_merge)
    to_split = resolve_split(split_weights, nabla)




end
