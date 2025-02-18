

abstract type MemoryModule end

function write! end

function read end

function maintain! end

struct AdaptiveMemory <:MemoryModule
    chain::APChain
    vtable::Dict{Int64, }
end

"""
Defines the granularity mapping for a current trace format
"""
struct GranularitySchema
    """Granularity for each object"""
    gs::Vector{UInt8}
    c::Int32
end

function maintain!(m::AdaptiveMemory)
    # Get object weights
    nabla = compute_task_relevance()

    # split-merge
    merge_weights = compute_merge_weights()
    to_merge = resolve_merge(merge_weights, nabla)

    split_weights = compute_split_weights(to_merge)
    to_split = resolve_split(split_weights, nabla)




end
