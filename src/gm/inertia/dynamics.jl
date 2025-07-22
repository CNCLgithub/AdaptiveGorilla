function object_bounds(wm::InertiaWM)
    @unpack area_width, area_height = wm
    xs = S2V(-0.5*area_width, 0.5*area_width)
    ys = S2V(-0.5*area_height, 0.5*area_height)
    (xs, ys)
end

function death_weight(wm::InertiaWM, st::InertiaState)
    # object_count(st) > wm.object_rate && !(isempty(singles)) ? wm.birth_weight : 0.0
    # isempty(singles) ? 0.0 : wm.birth_weight
    object_count(st) <= wm.object_rate || isempty(st.singles) ?
        0.0 : wm.birth_weight
end

function death_from_switch(prev, idx)
    singles = PersistentVector(prev)
    idx == 0 && return singles
    new_singles = PersistentVector{InertiaSingle}()
    for (i, s) = enumerate(singles)
        i == idx && continue
        new_singles = FunctionalCollections.push(new_singles, s)
    end
    return new_singles
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
    FunctionalCollections.push(singles, baby)
end

function force_prior(o::InertiaSingle, wm::InertiaWM)
    @unpack stability, force_low, force_high = wm
    (stability, force_low, force_high)
end

function force_prior(e::InertiaEnsemble, wm::InertiaWM)
    @unpack stability, force_low, force_high = wm
    @unpack rate = e
    unstable = 1.0 - stability
    # More stable, the more objects
    w = 1.0 - unstable^rate
    (w,
     10 * force_low,
     20 * force_high)
end

function step(wm::InertiaWM,
              state::InertiaState,
              supdates::AbstractVector{S2V},
              eupdates::AbstractVector{S3V})
    @unpack singles, ensembles = state
    step(wm, singles, ensembles, supdates, eupdates)
end

function step(wm::InertiaWM,
              singles::AbstractVector{InertiaSingle},
              ensembles::AbstractVector{InertiaEnsemble},
              supdates::AbstractVector{S2V},
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
        # force accumulator
        facc = MVector{2, Float64}(supdates[i])
        # interactions with walls
        for w in walls
            force!(facc, wm, w, obj)
        end
        new_singles[i] = update_state(obj, wm, facc)
    end
    @inbounds for i = 1:ne
        new_ensembles[i] = update_state(ensembles[i], wm,
                                        eupdates[i])
    end
    # println("step ns=$(length(new_singles))")
    InertiaState(PersistentVector(new_singles),
                 PersistentVector(new_ensembles))
end

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
    update_state(::Object, ::GM, ::MVector{2, Float64})

resolve force on object
"""
function update_state end

function update_state(s::InertiaSingle, wm::InertiaWM, f::MVector{2, Float64})
    # treating force directly as velocity;
    # update velocity by x percentage;
    # but f isn't normalized to be similar to v
    @unpack mat, pos, vel, size = s
    @unpack area_height, area_width = wm
    # vx, vy = f
    vx, vy = vel + f
    mxv = 2.0 * wm.vel
    vx = clamp(vx, -mxv, mxv)
    vy = clamp(vy, -mxv, mxv)
    x = clamp(pos[1] + vx,
              -area_width * 0.5,
              area_width * 0.5)
    y = clamp(pos[2] + vy,
              -area_height * 0.5,
              area_height * 0.5)
    new_pos = S2V(x, y)
    new_vel = S2V(vx, vy)
    # @show (vel, f, new_vel)
    InertiaSingle(mat, new_pos, new_vel, size)
end


function update_state(e::InertiaEnsemble, wm::InertiaWM, update::S3V)
    iszero(e.rate) && return e
    bx, by = wm.dimensions
    @unpack pos, vel, var, rate = e
    f = S2V(update[1], update[2])
    dx, dy = f
    # mxv = 2. * wm.vel / sqrt(rate)
    # dx = clamp(dx, -mxv, mxv)
    # dy = clamp(dy, -mxv, mxv)
    new_vel = S2V(dx, dy)
    x, y = pos
    new_pos = S2V(clamp(x + dx, -0.5 * bx, 0.5 * bx),
                  clamp(y + dy, -0.5 * by, 0.5 * by))
    new_var = clamp(var * (1.0 + update[3]), 10.0, 1000.0)
    # @show (var, update[3])
    setproperties(e; pos = new_pos,
                  vel = new_vel,
                  var = new_var)
end
