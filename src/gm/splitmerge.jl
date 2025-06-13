"""
A granularity transformation
"""
abstract type GranularityMove end

"""
$(TYPEDEF)

Split an ensemble.
---
$(TYPEDFIELDS)
"""
struct SplitMove <: GranularityMove
    "Ensemble index"
    x::Int64
end

"""
$(TYPEDEF)

Merge two object representations into an ensemble.

Both representations do not need to be the same granularity.
---
$(TYPEDFIELDS)
"""
struct MergeMove <: GranularityMove
    "Global object index"
    a::Int64
    "Global object index"
    b::Int64
end

const SplitMergeMove = Union{SplitMove, MergeMove}


"""

    apply_granularity_move(m::SplitMergeMove, wm::WorldModel, state,...)

Applies a granularity move to a given state, returning a new state.

---

# Methods

$(METHODLIST)

"""
function apply_granularity_move end
