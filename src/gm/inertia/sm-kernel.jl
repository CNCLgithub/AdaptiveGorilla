# Given an current state, S, propose a set of merges


# S_1 = [a,b,c]
# S_2 = [M(a, b), c]
# S_2 = [M(a, b), M(a, c)] = [M(a,b,c)]
# [M(a, b, c)]

# M(a::InertiaObject, b::InertiaObject) -> c::InertiaObject

