export wm_inertia



################################################################################
# Object state prior
################################################################################
@gen static function state_prior(wm::InertiaWM)

    xs, ys = object_bounds(wm)
    x = @trace(uniform(xs[1], xs[2]), :x)
    y = @trace(uniform(ys[1], ys[2]), :y)

    ang = @trace(uniform(0., 2*pi), :ang)
    mag = @trace(normal(wm.vel, 1e-2), :std)

    pos = S2V(x, y)
    vel = S2V(mag*cos(ang), mag*sin(ang))

    result::Tuple{S2V, S2V} = (pos, vel)
    return result
end

################################################################################
# Birth
################################################################################

@gen static function birth_single(wm::InertiaWM)
    ms = materials(wm)
    nm = length(ms)
    mws = Fill(1.0 / nm, nm)
    midx = @trace(categorical(mws), :material)
    material = ms[midx]
    loc, vel = @trace(state_prior(wm), :state)
    baby::InertiaSingle = InertiaSingle(material, loc, vel)
    return baby
end

# NOTE: Members of the switch have to be the same type of Gen.GM
# (e.g., both have to be dynamic or both static)

@gen function give_birth(wm::InertiaWM)
    ms = materials(wm)
    nm = length(ms)
    mws = Fill(1.0 / nm, nm)
    midx = @trace(categorical(mws), :material)
    material = ms[midx]

    xs, ys = object_bounds(wm)
    x = @trace(uniform(xs[1], xs[2]), :x)
    y = @trace(uniform(ys[1], ys[2]), :y)

    ang = @trace(uniform(0., 2*pi), :ang)
    mag = @trace(normal(wm.vel, 1.0), :std)

    loc = S2V(x, y)
    vel = S2V(mag*cos(ang), mag*sin(ang))

    result::InertiaSingle = InertiaSingle(material, loc, vel)
    return result
end

@gen function no_birth(wm::InertiaWM)
    # result::Vector{InertiaSingle} = InertiaSingle[]
    result::InertiaSingle = InertiaSingle(Light, S2V(0, 0), S2V(0, 0))
    return result
end

birth_switch = Gen.Switch(give_birth, no_birth)
# birth_switch = Gen.Switch(birth_single, no_birth)

@gen function birth_process(wm::InertiaWM, prev::InertiaState)
    w = birth_weight(wm, prev)
    pregnant ~ bernoulli(w)
    switch_idx = pregnant ? 1 : 2
    # potentially empty
    birth ~ birth_switch(switch_idx, wm)

    result::PersistentVector{InertiaSingle} =
        add_baby_from_switch(prev, birth, switch_idx)
    return result
end

@gen static function birth_ensemble(wm::InertiaWM, rate::Float64)
    # assuming | materials | = 2
    # NOTE: could replace with dirichlet
    plight ~ beta(0.5, 0.5)
    mws = S2V(plight, 1.0 - plight)
    loc, vel = @trace(state_prior(wm), :state)
    spread ~ inv_gamma(wm.ensemble_shape, wm.ensemble_scale)
    var = ensemble_var(wm, spread)
    ensemble::InertiaEnsemble =
        InertiaEnsemble(rate, mws, loc, var, vel)
    return ensemble
end

################################################################################
# Prior over initial state
################################################################################

@gen static function inertia_init(wm::InertiaWM)
    n ~ poisson(wm.object_rate) # total number of objects
    singles ~ Gen.Map(birth_single)(Fill(wm, n))
    state::InertiaState = InertiaState(wm, singles)
    return state
end


################################################################################
# Split-merge kernel
################################################################################

@gen static function sample_split(wm::InertiaWM, ens::InertiaEnsemble)
    ms = materials(wm)
    midx = @trace(categorical(ens.matws), :material)
    material = ms[midx]
    (ex, ey) = get_pos(ens)
    x ~ normal(ex, sqrt(ens.var))
    y ~ normal(ey, sqrt(ens.var))
    (vx, vy) = get_vel(ens)
    ang ~ von_mises(atan(vy, vx), 10.0)
    mag ~ normal(norm(ens.vel), wm.force_low)
    loc = S2V(x, y)
    vel = S2V(mag*cos(ang), mag*sin(ang))
    result::InertiaSingle = InertiaSingle(material, loc, vel)
    return result
end

@gen static function sample_split_pair(wm::InertiaWM, ens::InertiaEnsemble)
    s1 ~ sample_split(wm, ens)
    s2 ~ sample_split(wm, ens)
    result = (s1, s2)
    return result
end

@gen static function sample_merge(st::InertiaState, x::Int64, y::Int64)
    a = object_from_idx(st, x)
    b = object_from_idx(st, y)
    w = merge_probability(a, b) # how similar are a,b ?
    to_merge::Bool = @trace(bernoulli(w), :to_merge)
    return to_merge
end

################################################################################
# Dynamics
################################################################################

@gen static function inertia_force(wm::InertiaWM, o::Object)
    stability, force_low, force_high = force_prior(o, wm)
    is_stable ~ bernoulli(stability)

    # large angular variance if not stable
    var = is_stable ? force_low : force_high
    fx ~ normal(0.0, var)
    fy ~ normal(0.0, var)
    # converting back to vector form
    force::S2V = S2V(fx, fy)
    return force
end

@gen static function inertia_ensemble(wm::InertiaWM,
                                      e::InertiaEnsemble)
    force ~ inertia_force(wm, e)
    spread ~ uniform(1.0 - wm.ensemble_var_shift,
                     1.0 + wm.ensemble_var_shift)
    result::S3V = S3V(force[1], force[2], spread)
    return result
end

@gen static function inertia_kernel(t::Int64,
                                    prev::InertiaState,
                                    wm::InertiaWM)

    # Gorilla
    # REVIEW: add Death?
    singles = @trace(birth_process(wm, prev), :birth)
    s2::InertiaState = InertiaState(singles, s1.ensembles)

    # Random nudges
    ns = length(singles)
    ne = length(s2.ensembles)
    forces ~ Gen.Map(inertia_force)(Fill(wm, ns), s2.singles)
    eshifts ~ Gen.Map(inertia_ensemble)(Fill(wm, ne), s2.ensembles)
    s3::InertiaState = step(wm, s2, forces, eshifts)

    # RFS observations
    es = predict(wm, s3)
    xs ~ DetectionRFS(es)
    return s3
end

const inertia_unfold = Gen.Unfold(inertia_kernel)

@gen static function no_sm(x::InertiaState, wm::InertiaWM)
    return x
end

@gen static function split_ensemble(wm::InertiaWM, ens::InertiaEnsemble)
    ms = materials(wm)
    midx = @trace(categorical(ens.matws), :material)
    material = ms[midx]
    (ex, ey) = get_pos(ens)
    x ~ normal(ex, sqrt(ens.var))
    y ~ normal(ey, sqrt(ens.var))
    (vx, vy) = get_vel(ens)
    ang ~ von_mises(atan(vy, vx), 10.0)
    mag ~ normal(norm(ens.vel), wm.force_low)
    loc = S2V(x, y)
    vel = S2V(mag*cos(ang), mag*sin(ang))
    result::InertiaSingle = InertiaSingle(material, loc, vel)
    return result
end

@gen static function inertia_split(st::InertiaState, wm::InertiaWM)
    ne = length(st.ensembles)
    idx ~ categorical(Fill(1.0 / ne, ne))
    split ~ split_ensemble(wm, st.ensembles[idx])
    # TODO: refactor `apply_granularity_move`
    result::InertiaState = apply_granularity_move(Split(idx), wm, st, split)
    return result
end

@gen static function inertia_merge(st::InertiaState, wm::InertiaWM)
    ntotal = length(st.singles) + length(st.ensembles)
    nmerges = ncr(ntotal, 2)
    # sample lexographic index of 2-comb
    pair ~ categorical(Fill(1.0 / nmerges, nmerges))
    a, b = combination(ntotal, 2, pair)
    # TODO: refactor `apply_granularity_move`
    result::InertiaState = apply_granularity_move(Merge(a, b), wm, st)
    return result
end

split_merge_switch = Switch(no_sm, inertia_split, inertia_merge)

@gen static function inertia_granularity(wm::InertiaWM, x::InertiaState)
    nsm ~ categorical([0.5, 0.25, 0.25]) # nothing - split - merge
    state ~ split_merge_switch(nsm, x, wm)
    return state
end

################################################################################
# Full models
################################################################################

@gen static function wm_inertia(k::Int, wm::InertiaWM, init_state::InertiaState)
    s0 ~ inertia_granularity(wm, init_state)
    kernel ~ inertia_unfold(k, s0, wm)
    result = (init_st, states)
    return result
end
