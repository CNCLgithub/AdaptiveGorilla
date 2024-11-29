# Given an current state, S, propose a set of merges


S_1 = [a,b,c]
S_2 = [M(a, b), c]
S_2 = [M(a, b), M(a, c)] = [M(a,b,c)]
[M(a, b, c)]

M(a::Object, b::Object) -> c::Object

@gen function merge_kernel(st::InertiaState, wm::InertiaWM)
    objs = all_objects(st)
    (pairs, merged) = merging ~ sample_merges(objs)
    result = reduce_merging(obs, pairs, merged)
end

@gen function sample_merges(xs::Vector{Object})
    pairs = pair_vec(xs)
    samples ~ Gen.Map(sample_merge)(pairs)
    return (pairs, samples)
 end

@gen function sample_merge(a::Object, b::Object)
    w = merge_probability(a, b) # how similar are a,b ?
    merge ~ bernoulli(w)
end


# TODO
function reduce_merging(xs::Vector, pairs::Vector, merged::Vector)
end

# TODO
function pair_vec(xs::Vector)
end

# TODO
function determ_merge(a::Object, b::Object)
end


# TODO
function merge_probability(a::Object, b::Object)
end
