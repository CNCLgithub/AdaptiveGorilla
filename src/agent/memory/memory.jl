

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
struct AdaptiveGranularity <: GranularityProtocal
    """Ratio between objectve 1 and 2"""
    m::Float64
    """Number of particles to approx split-merge alpha"""
    alpha_steps::Int64
    """Number of objects to evaluate"""
    object_steps::Int64
end

function adapt_granularity!(chain::APChain, p::AdaptiveProtocol, g::AdaptiveGranularity)
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
