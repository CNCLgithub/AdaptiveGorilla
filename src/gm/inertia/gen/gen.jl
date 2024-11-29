export wm_inertia

include("init.jl")


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
    var ~ inv_gamma(wm.ensemble_shape, wm.ensemble_scale)
    ensemble::InertiaEnsemble =
        InertiaEnsemble(rate, mws, loc, var, vel)
    return ensemble
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

    (singles, ensembles) ~ merge_kernel(prev, wm)

    # REVIEW: add Death?
    # Birth
    singles = @trace(birth_process(wm, prev), :birth)
    n = length(singles)

    # Random nudges
    forces ~ Gen.Map(inertia_force)(Fill(wm, n), singles)
    eshift ~ inertia_ensemble(wm, prev.ensemble)
    next::InertiaState = step(wm, singles, prev.ensemble,
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
