# Given an current state, S, propose a set of merges


# S_1 = [a,b,c]
# S_2 = [M(a, b), c]
# S_2 = [M(a, b), M(a, c)] = [M(a,b,c)]
# [M(a, b, c)]

# M(a::InertiaObject, b::InertiaObject) -> c::InertiaObject

abstract type GranularityMove end

struct SplitMove <: GranularityMove
    x::Int64
end

struct MergeMove <: GranularityMove
    a::Int64
    b::Int64
end

const SplitMergeMove = Union{SplitMove, MergeMove}
