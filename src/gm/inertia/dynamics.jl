function object_bounds(wm::InertiaWM)
    @unpack area_width, area_height = wm
    xs = S2V(-0.5*area_width, 0.5*area_width)
    ys = S2V(-0.5*area_height, 0.5*area_height)
    (xs, ys)
end

"""
Uniform probability of Pr(Obj | Death = true)
"""
function death_weights(wm::InertiaWM, st::InertiaState)
    n = length(st.singles)
    fill(1.0 / n, n)
end

function death_weight(wm::InertiaWM, st::InertiaState)
    object_count(st) > wm.object_rate && !isempty(st.singles) ?
        0.01 : 0
        # wm.birth_weight : 0
        # (1.0 - wm.birth_weight) : 0
end

function death_from_switch(prev, idx)
    # do nothing
    idx == 0 && return prev
    new_singles = prev.singles[[1:idx-1; idx+1:end]]
    return InertiaState(new_singles, prev.ensembles)
end

function birth_weight(wm::InertiaWM, st::InertiaState)
    @unpack singles, ensembles = st
    # length(singles) + sum(rate, ensembles; init=0.0) <= wm.object_rate ?
    #     wm.birth_weight : 0.0
    object_count(st) > wm.object_rate ?
        0.0 : wm.birth_weight
end

function add_baby_from_switch(prev, baby)
    singles = PersistentVector(prev.singles)
    singles = FunctionalCollections.push(singles, baby)
    InertiaState(singles, prev.ensembles)
end

function force_prior(o::InertiaSingle, wm::InertiaWM)
    @unpack stability, force_low, force_high = wm
    (stability, force_low, force_high)
end

function step(wm::InertiaWM,
              state::InertiaState,
              supdates::AbstractVector{S3V},
              eupdates::AbstractVector{S3V})
    @unpack singles, ensembles = state
    step(wm, singles, ensembles, supdates, eupdates)
end

function step(wm::InertiaWM,
              singles::AbstractVector{InertiaSingle},
              ensembles::AbstractVector{InertiaEnsemble},
              supdates::AbstractVector{S3V},
              eupdates::AbstractVector{S3V})
    @unpack walls = wm
    ns = length(singles)
    new_singles = Vector{InertiaSingle}(undef, ns)
    @assert ns == length(supdates) "$(length(supdates)) updates but only $ns singles"
    ne = length(ensembles)
    new_ensembles = Vector{InertiaEnsemble}(undef, ne)
    @assert ne == length(eupdates) "$(length(eupdates)) updates but only $ne ensembles"
    @inbounds for i = 1:ns
        obj = singles[i]
        new_singles[i] = update_state(obj, wm, supdates[i])
    end
    @inbounds for i = 1:ne
        new_ensembles[i] = update_state(ensembles[i], wm,
                                        eupdates[i])
    end
    # println("step ns=$(length(new_singles))")
    InertiaState(PersistentVector(new_singles),
                 PersistentVector(new_ensembles))
end

# TODO: Depricate `force`

"""Computes the force of A -> B"""
function force!(f::MVector{2, Float64}, ::InertiaWM, ::Object, ::Object)
    return nothing
end

function force!(f::MVector{2, Float64}, wm::InertiaWM, w::Wall, s::InertiaSingle)
    pos = get_pos(s)
    @unpack wall_rep_m, wall_rep_a, wall_rep_x0 = wm
    n = LinearAlgebra.norm(w.normal .* pos + w.nd)
    f .+= wall_rep_m * exp(-1 * (wall_rep_a * (n - wall_rep_x0))) * w.normal
    return nothing
end

# REVIEW: wall -> ensemble force
function force!(f::MVector{2, Float64}, wm::InertiaWM, w::Wall, e::InertiaEnsemble)
    # @unpack pos = d
    # @unpack wall_rep_m, wall_rep_a, wall_rep_x0 = dm
    # n = LinearAlgebra.norm(w.normal .* pos + w.nd)
    # f .+= wall_rep_m * exp(-1 * (wall_rep_a * (n - wall_rep_x0))) * w.normal
    return nothing
end

"""
    update_state(::Object, ::GM, ::MVector{3, Float64})

resolve force on object
"""
function update_state end

function update_state(s::InertiaSingle, wm::InertiaWM, f::S3V)
    # treating force directly as velocity;
    # update velocity by x percentage;
    # but f isn't normalized to be similar to v
    @unpack area_height, area_width = wm
    @unpack mat, pos, vel, avel = s

    tvx, tvy = vel
    dvx, dvy, dvo = f
    
    new_avel = clamp(avel + dvo, -pi/5, pi/5)
    cos_om = cos(new_avel)
    sin_om = sin(new_avel)
    avx = cos_om * tvx - sin_om * tvy
    avy = sin_om * tvx + cos_om * tvy
    
    mxv = 2.0 * wm.vel
    vx = clamp(avx + dvx, -mxv, mxv)
    vy = clamp(avy + dvy, -mxv, mxv)

    x = clamp(pos[1] + vx,
              -area_width * 0.5,
              area_width * 0.5)
    y = clamp(pos[2] + vy,
              -area_height * 0.5,
              area_height * 0.5)
    new_pos = S2V(x, y)
    new_tvel = S2V(vx, vy)
    InertiaSingle(mat, new_pos, new_tvel, new_avel)
end


function update_state(e::InertiaEnsemble, wm::InertiaWM, update::S3V)
    iszero(e.rate) && return e
    @unpack pos, vel, var, rate = e
    x, y = pos
    dx, dy, dvar = update

    bx, by = wm.dimensions
    new_pos = S2V(clamp(x + dx, -0.5 * bx, 0.5 * bx),
                  clamp(y + dy, -0.5 * by, 0.5 * by))
    new_var = clamp(var + dvar, 10.0, bx)
    setproperties(e; pos = new_pos,
                  vel = S2V(dx, dy),
                  var = new_var)
end
