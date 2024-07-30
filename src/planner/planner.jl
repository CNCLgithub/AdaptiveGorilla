export Planner,
    plan,
    CollisionPlanner,
    energy

abstract type Planner{T} end

function plan end

@with_kw struct CollisionPlanner <: Planner{InertiaTrace}
    mat::Material
    tol::Float64 = 1E-4
end

function plan(pl::CollisionPlanner, tr::InertiaTrace)
    _, wm = get_args(tr)
    _, states = get_retval(tr)
    state = last(states)
    plan(pl, state, wm)
end

function plan(pl::CollisionPlanner, state::InertiaState, wm::InertiaWM)
    @unpack mat, tol = pl
    @unpack walls = wm
    @unpack singles, ensemble = state
    e = 0.0
    @inbounds for j = 1:length(singles)
        _e = 0.0
        single = singles[j]
        single.mat == mat || continue
        for k = 1:4
            # display(walls[k])
            _e += energy(walls[k], single)
            # e = logsumexp(e, _e)
        end
        # @show j => get_pos(single) =>  _e
        e += _e
    end
    # TODO: energy for ensembles
    return e
end

function energy(w::Wall, s::InertiaSingle)
    pos = get_pos(s)
    vel = get_vel(s)
    # REVIEW: energy - incorporate speed?
    # speed of approach
    # rate = sum(w.normal .* vel)
    # rate = max(1E-5, rate)
    # distance to wall
    # dis = norm(w.normal .* pos - w.nd) - s.size
    dis = w.d - sum(w.normal .* pos) # - s.size
    dis = max(1., dis)
    # @show dis
    e = 100.0 / abs(dis)
    # max(0.0, log(dis) - log(rate))
    e
end
