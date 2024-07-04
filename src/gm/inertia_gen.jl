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

@gen static function give_birth(wm::InertiaWM)
    baby ~ birth_single(wm)
    result::Vector{InertiaSingle} = [baby]
    return result
end

@gen static function no_birth(wm::InertiaWM)
    result::Vector{InertiaSingle} = InertiaSingle[]
    return result
end

birth_switch = Gen.Switch(give_birth, no_birth)

# Must wrap switch combinator to keep track of changes in `i`
@gen function birth_or_not(i::Int, wm::InertiaWM)
    to_add::Vector{InertiaSingle} = @trace(birth_switch(i, wm),
                                           :birth_switch)
    return to_add
end

@gen function birth_process(wm::InertiaWM, prev::InertiaState)
    pregnant ~ bernoulli(wm.birth_weight)
    switch_idx = pregnant ? 1 : 2
    # potentially empty
    birth ~ birth_or_not(switch_idx, wm)
    result::PersistentVector{InertiaSingle} = append(prev.singles, birth)
    return result
end

@gen static function birth_ensemble(wm::InertiaWM, rate::Float64)
    ms = materials(wm)
    nm = length(ms)
    mws = Fill(1.0 / nm, nm)
    midx = @trace(categorical(mws), :material)
    material = ms[midx]
    loc, vel = @trace(state_prior(wm), :state)
    var ~ inv_gamma(wm.ensemble_shape, wm.ensemble_scale)
    ensemble::InertiaEnsemble =
        InertiaEnsemble(rate, material, loc, var, vel)
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

@gen static function inertia_kernel(t::Int64,
                                    prev::InertiaState,
                                    wm::InertiaWM)
    # REVIEW: add Death?
    # Birth
    singles = @trace(birth_process(wm, prev), :birth)
    n = length(singles)

    # Random nudges
    forces ~ Gen.Map(inertia_force)(Fill(wm, n), singles)
    fens ~ inertia_force(wm, prev.ensemble)
    next::InertiaState = step(wm, singles, prev.ensemble, prev.walls,
                              forces, fens)

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
