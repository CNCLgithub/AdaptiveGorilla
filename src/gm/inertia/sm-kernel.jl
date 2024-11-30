# Given an current state, S, propose a set of merges


# S_1 = [a,b,c]
# S_2 = [M(a, b), c]
# S_2 = [M(a, b), M(a, c)] = [M(a,b,c)]
# [M(a, b, c)]

# M(a::InertiaObject, b::InertiaObject) -> c::InertiaObject

@gen function merge_kernel(st::InertiaState, wm::InertiaWM)
    objs = all_objects(st)
    (pairs, merged) = merging ~ sample_merges(objs)
    result = reduce_merging(obs, pairs, merged)
end

@gen function sample_merges(xs::Vector{InertiaObject})
    pairs = pair_vec(xs)
    samples ~ Gen.Map(sample_merge)(pairs)
    return (pairs, samples)
 end

@gen function sample_merge(a::InertiaObject, b::InertiaObject)
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
function determ_merge(a::InertiaObject, b::InertiaObject)
end

function merge_probability(a::InertiaSingle, b::InertiaSingle)
    color = a.mat === b.mat ? 1.0 : 0.0
    l2 = norm(get_pos(a) - get_pos(b))
    ca = vec2_angle(get_vel(a), get_vel(b))
    color * 0.5 * exp(-(l2 * ca) / max(a.size, b.size))
end

function merge_probability(a::InertiaSingle, b::InertiaEnsemble)
    color = b.matws[Int64(a.mat)]
    l2 = norm(get_pos(a) - get_pos(b)) / sqrt(b.var)
    ca = vec2_angle(get_vel(a), get_vel(b))
    color * 0.5 * exp(-(l2 * ca) / max(a.size, sqrt(b.var)))
end

function merge_probability(a::InertiaEnsemble, b::InertiaSingle)
    merge_probability(b, a)
end

function merge_probability(a::InertiaEnsemble, b::InertiaEnsemble)
    color = norm(a.matws - b.matws)
    l2 = norm(get_pos(a) - get_pos(b)) / (sqrt(b.var) + sqrt(a.var))
    ca = vec2_angle(get_vel(a), get_vel(b))
    0.5 * exp(-(l2 * ca * color) / sqrt(max(a.var, b.var)))
end
