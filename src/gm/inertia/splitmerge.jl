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

VAR_POS = 1.0 / 3.0

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
    # var = get_var(e) - (norm(delta_pos) - get_var(e)) / new_count
    var = get_var(e) - VAR_POS * norm(delta_pos) / new_count
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
    delta_pos = (get_pos(a) - get_pos(b))
    new_pos = 0.5 .* (get_pos(a) + get_pos(b))
    mag_pos = max(norm(delta_pos), 1.0)
    std = VAR_POS * mag_pos
    new_vel = 0.5 * (get_vel(a) + get_vel(b))
    InertiaEnsemble(
        2,
        matws,
        new_pos,
        std,
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
    delta_pos = get_pos(a) - get_pos(b)
    delta_vel = get_vel(a) - get_vel(b)
    new_pos = get_pos(b) + delta_pos / new_count
    new_vel = get_vel(b) + delta_vel / new_count
    var = get_var(b) + (VAR_POS*norm(delta_pos)) / new_count
    # println("Growing λ=$(b.rate)+1 $(get_var(b)) -> $(var)")
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
    new_pos = (a.rate/new_count)*get_pos(a) +
        (b.rate/new_count)*get_pos(b)
    new_vel = (a.rate/new_count)*get_vel(a) +
        (b.rate/new_count)*get_vel(b)

    delta_pos = get_pos(a) - get_pos(b)
    mag_pos = max(norm(delta_pos), 1.0)
    var = VAR_POS*mag_pos + get_var(a) * a.rate / new_count +
        get_var(b) * b.rate / new_count
    # println("Growing $(get_var(a)), $(get_var(b)) -> $(var)")
    InertiaEnsemble(
        new_count,
        matws,
        new_pos,
        var,
        new_vel
    )
end

function split_merge_weights(wm::InertiaWM, x::InertiaState)
    # Nothing | Split | Merge
    ws = [1.0, 1.0, 1.0]
    @unpack singles, ensembles = x
    if isempty(ensembles)
        ws[2] = 0.0
    end
    if length(singles) + length(ensembles) < 2
        ws[3] = 0.0
    end
    lmul!(1.0 / sum(ws), ws)
    return ws
end
