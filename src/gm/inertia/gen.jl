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
    mag = @trace(normal(wm.vel, 1e-2), :std)

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
    pregnant ~ bernoulli(wm.birth_weight)
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
    var = spread * (min(wm.area_width, wm.area_height))
    ensemble::InertiaEnsemble =
        InertiaEnsemble(rate, mws, loc, var, vel)
    return ensemble
end

################################################################################
# Prior over initial state
################################################################################

@gen static function inertia_init(wm::InertiaWM)
    n ~ poisson(wm.object_rate) # total number of objects
    k ~ binom(n, wm.irate) # number of individuals
    wms = Fill(wm, k)
    singles ~ Gen.Map(birth_single)(wms)
    ensemble ~ birth_ensemble(wm, n - k) 
    state::InertiaState = InertiaState(wm, singles, ensemble)
    return state
end


################################################################################
# Split-merge kernel
################################################################################

@gen function split_kernel(st::InertiaState, wm::InertiaWM)
    w = split_probability(st, wm)
    split ~ bernoulli(w)
    sidx = split ? 1 : 2
    birth ~ birth_switch(sidx, wm)
    result::InertiaState = maybe_apply_split(st, birth, split)
    return result
end

@gen static function sample_merge(a::InertiaObject, b::InertiaObject)
    w = merge_probability(a, b) # how similar are a,b ?
    to_merge::Bool = @trace(bernoulli(w), :to_merge)
    return to_merge
end

@gen static function merge_kernel(st::InertiaState, wm::InertiaWM)
    n = length(st.singles)
    mergers ~ Gen.Map(sample_merge)(st.singles, Fill(st.ensemble, n))
    result::InertiaState = apply_mergers(st, mergers)
    return result
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

    merge ~ merge_kernel(prev, wm)
    sm = @trace(split_kernel(merge, wm), :split)

    # REVIEW: add Death?
    # Birth
    singles = @trace(birth_process(wm, sm), :birth)
    n = length(singles)

    # Random nudges
    forces ~ Gen.Map(inertia_force)(Fill(wm, n), singles)
    eshift ~ inertia_ensemble(wm, sm.ensemble)
    next::InertiaState = step(wm, singles, sm.ensemble,
                              forces, eshift)

    # predict observations as a random finite set
    es = predict(wm, next)
    xs ~ DetectionRFS(es)

    return next
end



################################################################################
# Full models
################################################################################

@gen static function wm_inertia(k::Int, wm::InertiaWM)
    init_st = @trace(inertia_init(wm), :init_state)
    states = @trace(Gen.Unfold(inertia_kernel)(k, init_st, wm), :kernel)
    result = (init_st, states)
    return result
end
