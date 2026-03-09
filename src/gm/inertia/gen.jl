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
    material ~ categorical(mws)

    xs, ys = object_bounds(wm)
    x ~ uniform(xs[1], xs[2])
    y ~ uniform(ys[1], ys[2])

    ang ~ uniform(0., 2*pi)
    mag ~ normal(wm.vel, 1.0)

    loc = S2V(x, y)
    vel = S2V(mag*cos(ang), mag*sin(ang))

    result::InertiaSingle = InertiaSingle(ms[material], loc, vel)
    return result
end

@gen function birth_process(wm::InertiaWM, prev::InertiaState)
    birth ~ give_birth(wm)
    result::InertiaState = add_baby_from_switch(prev, birth)
    return result
end

@gen function death_process(wm::InertiaWM, prev::InertiaState)
    ws = death_weights(wm, prev)
    dead ~ categorical(ws)
    result::InertiaState = death_from_switch(prev, dead)
    return result
end

@gen function no_birth_death(wm::InertiaWM, prev::InertiaState)
    result::InertiaState = prev
    return result
end

birth_death_switch = Gen.Switch(no_birth_death, birth_process, death_process)

@gen function birth_death_process(wm::InertiaWM, prev::InertiaState)
    bw = birth_weight(wm, prev)
    dw = death_weight(wm, prev)
    ws = [1.0, bw, dw]
    lmul!(1.0 / sum(ws), ws)
    i ~ categorical(ws)
    switch ~ birth_death_switch(i, wm, prev)
    bd::InertiaState = switch
    return bd
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
    ang_var = is_stable ? wm.ang_acc / 10.0 : wm.ang_acc
    fa ~ normal(0.0, ang_var)
    # converting back to vector form
    force::S3V = S3V(fx, fy, fa)
    return force
end

@gen static function inertia_ensemble(wm::InertiaWM)
    fx ~ normal(0.0, wm.ensemble_force)
    fy ~ normal(0.0, wm.ensemble_force)
    spread ~ uniform(-wm.ensemble_var_shift, wm.ensemble_var_shift)
    result::S3V = S3V(fx, fy, spread)
    return result
end

@gen static function inertia_kernel(t::Int64,
                                    prev::InertiaState,
                                    wm::InertiaWM)
    # birth-death
    bd ~ birth_death_process(wm, prev)

    # Random nudges
    ns = length(bd.singles)
    ne = length(bd.ensembles)
    forces ~ Gen.Map(inertia_force)(Fill(wm, ns), bd.singles)
    eshifts ~ Gen.Map(inertia_ensemble)(Fill(wm, ne))
    shifted::InertiaState = step(wm, bd, forces, eshifts)

    # RFS observations
    es = predict(wm, shifted)
    xs ~ DetectionRFS(es)
    return shifted
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
    result::InertiaState = apply_granularity_move(SplitMove(idx), wm, st, split)
    return result
end

@gen static function inertia_merge(st::InertiaState, wm::InertiaWM)
    ntotal = length(st.singles) + length(st.ensembles)
    nmerges = ncr(ntotal, 2)
    # sample lexographic index of 2-comb
    pair ~ categorical(Fill(1.0 / nmerges, nmerges))
    a, b = combination(ntotal, 2, pair)
    result::InertiaState = apply_granularity_move(MergeMove(a, b), wm, st)
    return result
end

split_merge_switch = Switch(no_sm, inertia_split, inertia_merge)

@gen static function inertia_granularity(wm::InertiaWM, x::InertiaState)
    ws = split_merge_weights(wm, x)
    nsm ~ categorical(ws) # 1 => nothing, 2 => split, 3 => merge
    state ~ split_merge_switch(nsm, x, wm)
    return state
end

################################################################################
# Full models
################################################################################

@gen static function wm_inertia(k::Int, wm::InertiaWM, init_state::InertiaState)
    s0 ~ inertia_granularity(wm, init_state)
    kernel ~ inertia_unfold(k, s0, wm)
    return kernel
end
