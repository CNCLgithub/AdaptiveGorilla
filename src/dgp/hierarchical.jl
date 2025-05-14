
using MOTCore: inbounds, deflect

@with_kw struct HWM <: WorldModel
    inner::SchollWM
    cohesion::Float64 = 1.0
    group_noise::Float64 = 1.0
end

inner(wm::HWM) = wm.inner

struct HWS <: WorldState{HWM}
    groups::Vector{Group}
    objects::Vector{Dot}
end

struct Group
    pos::S2V
    vel::S2V
    members::Vector{Int64}
end

get_pos(g::Group) = g.pos
get_vel(g::Group) = g.vel
members(g::Group) = g.members

function step(wm::HWM, ws::HWS,
              group_deltas::Vector{S2V},
              object_deltas::Vector{S2V})
    # Setup
    @unpack groups, objects = ws
    ng = length(groups)
    no = length(objects)
    new_groups = Vector{Group}(undef, ng)
    new_objects = Vector{Dot}(undef, no)
    # Update groups
    @inbounds for i = 1:ng
        group = groups[i]
        delta = group_deltas[i]
        new_groups[i] = update_elem(group, wm, delta)
        for m = members(group)
            object_deltas[m] = object_deltas[m] + delta
        end
    end
    # Update dots
    @inbounds for i = 1:no
        obj = objects[i]
        delta = object_deltas[i]
        new_objects[i] = update_elem(obj, wm, delta)
    end
    # Return updates elements
    HWS(new_groups, new_objects)
end

function update_elem(e::Group, wm::HWM, delta::S2V)
    inner_wm = inner(wm)
    pos = get_pos(e)
    vel = get_vel(e)
    radius = farthest_member(e)
    new_pos, new_vel = safe_update_state(pos, vel, radius,
                                         delta,
                                         inner_wm)
    Group(new_pos, new_vel, e.members)
end

function update_elem(e::Dot, wm::HWM, delta::S2V)
    inner_wm = inner(wm)
    pos = get_pos(e)
    vel = get_vel(e)
    new_pos, new_vel = safe_update_state(pos, vel,
                                         e.radius,
                                         delta,
                                         inner_wm)
    Dot(e.radius, new_pos, new_vel)
end

function safe_update_state(pos::S2V, vel::S2V, radius::Float64,
                           delta::S2V, wm::SchollWM)
    new_vel = safe_update_vel(vel, delta, wm)
    new_pos = safe_update_pos(pos, new_vel, radius, wm)
    (new_pos, new_vel)
end

function safe_update_vel(vel::S2V, delta::S2V, wm::SchollWM)
    @unpack vel_min, vel_max = wm
    # random delta for velocity
    new_vel = vel + delta
    # clamp velocity
    nv = normalize(new_vel)
    mag = clamp(norm(new_vel), vel_min, vel_max)
    nv .* mag
end

function safe_update_pos(pos::S2V, vel::S2V, radius::Float64,
                         wm::SchollWM)
    @unpack area_width, area_height = wm
    # adjust out of bounds positions
    new_pos = pos + new_vel
    if !in_bounds(wm, new_pos)
        new_vel = deflect(wm, pos, new_vel)
        new_pos = pos + new_vel
    end
    new_pos = S2V(
        clamp(new_pos[1], -area_width * 0.5 + radius,
              area_width * 0.5  - radius),
        clamp(new_pos[2], -area_height * 0.5 + radius,
              area_height * 0.5  - radius)
    )
end

################################################################################
# Gen
################################################################################

@gen (static) function scholl_delta(wm::SchollWM)
    flip = @trace(bernoulli(wm.vel_prob), :flip)
    fx = @trace(normal(0, wm.vel_step), :fx)
    fy = @trace(normal(0, wm.vel_step), :fy)
    f::SVector{2, Float64} = SVector{2, Float64}(fx * flip,
                                                 fy * flip)
    return f
end

@gen (static) function scholl_kernel(t::Int,
                                        prev_st::SchollState,
                                        wm::SchollWM)
    deltas = @trace(Gen.Map(scholl_delta)(Fill(wm, wm.n_dots)), :deltas)
    next_st::SchollState = step(wm, prev_st, deltas)
    return next_st
end

const scholl_chain = Gen.Unfold(scholl_kernel)


@gen (static) function wm_hwm(k::Int, wm::HWM)
    init_state = @trace(scholl_init(wm), :init_state)
    states = @trace(scholl_chain(k, init_state, wm), :kernel)
    result = (init_state, states)
    return result
end


################################################################################
# Misc
################################################################################

function farthest_member(g::Group, state::HWS)
    d = 0.0
    a = get_pos(g)
    for midx = members(g)
        dot = state.objects[midx]
        d = max(norm(get_pos(dot) - a), d)
    end
    return d
end
