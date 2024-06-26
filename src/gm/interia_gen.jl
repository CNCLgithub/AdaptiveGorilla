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

@gen static function birth_object(wm::InertiaWM)
    ms = materials(wm)
    nm = length(ms)
    mws = Fill(1.0 / nm, nm)
    midx = @trace(categorial(mws), :material)
    material = ms[midx]
    loc, vel = @trace(state_prior(wm), :state)
    baby::InertiaObject = InertiaObject(material, loc, vel)
    return baby
end

@gen static function give_birth(wm::InertiaWM)
    baby ~ birth_object(wm)
    result::Vector{InertiaObject} = [baby]
    return result
end

@gen static function no_birth(wm::InertiaWM)
    result::Vector{InertiaObject} = InertiaObject[]
    return result
end

birth_switch = Gen.Switch(give_birth, no_birth)

# Must wrap switch combinator to keep track of changes in `i`
@gen function birth_or_not(i::Int)
    to_add::Vector{InertiaObject} = @trace(birth_switch(i), :birth_switch)
    return to_add
end

@gen function birth_process(wm::InertiaWM, prev::InertiaState)
    birth = @trace(bernoulli(wm.birth_weight), :give_birth)
    switch_idx = birth ? 1 : 2
    # potentially empty
    baby ~ birth_or_not(switch_idx)
    result::PersistentVector{InertiaObject} = push(prev.objects, baby)
    return result
end

@gen static function birth_ensemble(wm::InertiaWM, rate::Float64)
    ms = materials(wm)
    nm = length(ms)
    mws = Fill(1.0 / nm, nm)
    midx = @trace(categorial(mws), :material)
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
    n ~ poisson(wm.object_rate)
    k ~ binomial(wm.irate, n)
    wms = Fill(wm, k)
    objects ~ Gen.Map(birth_object)(wms)
    ensemble ~ birth_ensemble(wm, n - k)
    state::InertiaState = InertiaState(wm, objects, ensemble)
    return state
end

################################################################################
# Dynamics
################################################################################

@gen static function inertia_force(wm::InertiaWM, o::Object)

    stability, ang_var, mag_var = force_weights(o, wm)

    is_stable = @trace(bernoulli(stability), :inertia)

    # large angular variance if not stable
    k = is_stable ? ang_var : 1.0
    ang = @trace(von_mises(0.0, k), :ang)

    w = is_stable ? mag_var : 10.0
    mag = @trace(normal(0.0, w), :mag)

    # converting back to vector form
    force::S2V = S2V(mag * cos(ang), mag * sin(ang))
    return force
end

@gen static function inertia_kernel(t::Int64,
                                    prev::InertiaState,
                                    wm::InertiaWM)


    # REVIEW: add Death?
    # Birth
    objects = @trace(birth_process(wm, prev), :birth)
    n = length(objects)

    # Random nudges
    updates = @trace(Gen.Map(inertia_step)(Fill(wm, n), objects), :force)

    next::InertiaState = next_state(wm, objects, updates)

    # predict observations as a random finite set
    es = predict(wm, t, next)
    xs = @trace(DetectionPMBRFS(es), :detections)

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
