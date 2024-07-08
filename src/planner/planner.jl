export Planner,
    plan,
    CollisionPlanner,
    energy

abstract type Planner end


function plan end

@with_kw struct CollisionPlanner
    mat::Material
    tol::Float64 = 1E-4
end

function plan(pl::CollisionPlanner, tr::InertiaTrace)
    _, states = get_retval(tr)
    state = last(states)
    plan(pl, state)
end

function plan(pl::CollisionPlanner, state::InertiaState)
    @unpack mat, tol = pl
    @unpack singles, ensembles, walls = state
    e = -Inf
    @inbounds for j = 1:length(singles)
        single = singles[j]
        single.mat == match || continue
        for k = 1:4
            e = logsumexp(e, energy(walls[k], single))
        end
    end
    return e
end

function energy(w::Wall, s::InertiaSingle)
    pos = get_pos(s)
    vel = get_vel(s)
    rate = sum(w.normal .* vel)
    rate = max(0.0, rate)
    # dis = norm(w.normal .* pos - w.nd) - s.size
    dis = w.d - sum(w.normal .* pos) - s.size
    dis = max(0.0, dis)
    @show dis
    @show rate
    min(0.0, -(log(dis) - log(rate)))
end
