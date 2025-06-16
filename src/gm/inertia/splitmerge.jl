"""
$(TYPEDEF)

Merges two object representations in `InertiaWM`
"""
function apply_granularity_move(m::MergeMove, wm::InertiaWM, state::InertiaState)
    @unpack singles, ensembles = state
    ns = length(singles)
    ne = length(ensembles)
    a = object_from_idx(state, m.a)
    b = object_from_idx(state, m.b)
    e = apply_merge(a, b)
    # need to determine what object types were merged
    delta_ns = 0
    delta_ns += isa(a, InertiaSingle) ? -1 : 0
    delta_ns += isa(b, InertiaSingle) ? -1 : 0
    delta_ne = 0
    if isa(a, InertiaSingle) && isa(b, InertiaSingle)
        delta_ne = 1

    elseif isa(a, InertiaEnsemble) && isa(b, InertiaEnsemble)
        delta_ne = -1
    end
    new_singles = Vector{InertiaSingle}(undef, ns + delta_ns)
    new_ensembles = Vector{InertiaEnsemble}(undef, ne + delta_ne)
    # Remove any merged singles
    c = 1
    for i = 1:ns
        (isa(a, InertiaSingle) && i == m.a) && continue
        (isa(b, InertiaSingle) && i == m.b) && continue
        new_singles[c] = singles[i]
        c += 1
    end
    # Remove merged ensembles
    c = 1
    for i = 1:ne
        ((isa(b, InertiaEnsemble) && i + ns == m.b)  ||
            (isa(a, InertiaEnsemble) && i + ns == m.a)) &&
            continue
        new_ensembles[c] = ensembles[i]
        c += 1
    end
    new_ensembles[c] = e
    InertiaState(new_singles, new_ensembles)
end

"""
$(TYPEDEF)

Emits an individual object (`splitted`) from a given ensemble.
"""
function apply_granularity_move(m::SplitMove, wm::InertiaWM, state::InertiaState,
                                splitted::InertiaSingle)
    @unpack singles, ensembles = state
    ns = length(singles)
    ne = length(ensembles)
    e = ensembles[m.x]
    new_e = apply_split(e, splitted)
    # if the ensemble is a pair, we get two individuals
    delta_ns = e.rate == 2 ? 2 : 1
    delta_ne = e.rate == 2 ? -1 : 0
    new_singles = Vector{InertiaSingle}(undef, ns + delta_ns)
    new_ensembles = Vector{InertiaEnsemble}(undef, ne + delta_ne)

    new_singles[1:ns] = singles
    if e.rate == 2 # E -> (S, S)
        new_singles[ns + 1] = collapse(new_e)
        new_singles[ns + 2] = splitted
        c = 1
        for i = 1:ne
            i == m.x && continue
            new_ensembles[c] = ensembles[i]
            c += 1
        end
    else # E -> (E', S)
        new_singles[ns + 1] = splitted
        new_ensembles[:] = ensembles[:]
        new_ensembles[m.x] = new_e
    end
    InertiaState(new_singles, new_ensembles)
end

function apply_split(e::InertiaEnsemble, x::InertiaSingle)
    new_count = e.rate - 1
    matws = deepcopy(e.matws)
    lmul!(e.rate, matws)
    matws[Int64(x.mat)] -= 1
    lmul!(1.0 / new_count, matws)
    delta_pos = (get_pos(e) - get_pos(x)) / new_count
    delta_vel = (get_vel(e) - get_vel(x)) / new_count
    new_pos = get_pos(e) + delta_pos
    new_vel = get_vel(e) + delta_vel
    var = get_var(e) - norm(delta_pos) - norm(delta_vel)
    var = max(10.0, var)
    InertiaEnsemble(
        new_count,
        matws,
        new_pos,
        var,
        new_vel
    )
end

function collapse(x::InertiaEnsemble)
    InertiaSingle(
        Material(argmax(x.matws)),
        get_pos(x),
        get_vel(x),
    )
end

function apply_merge(a::InertiaSingle, b::InertiaSingle)
    matws = zeros(NMAT)
    matws[Int64(a.mat)] += 0.5
    matws[Int64(b.mat)] += 0.5
    new_pos = 0.5 .* (get_pos(a) + get_pos(b))
    # REVIEW: dampen vel based on count?
    # [2025-06-13 Fri] switched factor from 0.5 -> 0.25
    # new_vel = 0.25 .* (get_vel(a) + get_vel(b))
    # [2025-06-16 Mon] switched to sqrt
    new_vel = 0.1 .* (get_vel(a) + get_vel(b))
    var = 3 * norm(get_pos(a) - get_pos(b))
    InertiaEnsemble(
        2,
        matws,
        new_pos,
        var,
        new_vel
    )
end

function apply_merge(a::InertiaSingle, b::InertiaEnsemble)
    new_count = b.rate + 1
    # materials
    matws = deepcopy(b.matws)
    lmul!(b.rate, matws)
    matws[Int64(a.mat)] += 1
    lmul!(1.0 / new_count, matws)
    # movement and variance
    delta_pos = (get_pos(a) - get_pos(b)) / new_count
    delta_vel = (get_vel(a) - get_vel(b)) / new_count
    new_pos = get_pos(b) + delta_pos
    new_vel = get_vel(b) + delta_vel
    var = get_var(b) + norm(delta_pos) + norm(delta_vel)
    InertiaEnsemble(
        new_count,
        matws,
        new_pos,
        var,
        new_vel
    )
end

function apply_merge(a::InertiaEnsemble, b::InertiaSingle)
    apply_merge(b, a)
end

function apply_merge(a::InertiaEnsemble, b::InertiaEnsemble)
    new_count = a.rate + b.rate
    matws = a.rate .* a.matws + b.rate .* b.matws
    lmul!(1.0 / new_count, matws)
    delta_pos = (get_pos(a) - get_pos(b)) / new_count
    delta_vel = (get_vel(a) - get_vel(b)) / new_count
    new_pos = get_pos(b) + delta_pos
    new_vel = get_vel(b) + delta_vel
    var = 0.5 * (a.var + b.var) + norm(delta_pos) + norm(delta_vel)
    InertiaEnsemble(
        new_count,
        matws,
        new_pos,
        var,
        new_vel
    )
end

function split_merge_weights(wm::InertiaWM, x::InertiaState)
    ws = [2.0, 1.0, 1.0]
    @unpack singles, ensembles = x
    if isempty(ensembles)
        ws[2] = 0.0
    end
    if isempty(singles)
        ws[3] = 0.0
    end
    lmul!(1.0 / sum(ws), ws)
    return ws
end
