# Given an current state, S, propose a set of merges


# S_1 = [a,b,c]
# S_2 = [M(a, b), c]
# S_2 = [M(a, b), M(a, c)] = [M(a,b,c)]
# [M(a, b, c)]

# M(a::InertiaObject, b::InertiaObject) -> c::InertiaObject

abstract type GranularityMove end

struct SplitMove <: GranularityMove
    "Ensemble index"
    x::Int64
end

struct MergeMove <: GranularityMove
    "Global object index"
    a::Int64
    "Global object index"
    b::Int64
end

const SplitMergeMove = Union{SplitMove, MergeMove}
