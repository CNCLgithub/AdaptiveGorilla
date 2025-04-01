export Planner,
    PlanningModule
    plan!,
    CollisionCounter,
    energy

abstract type PlanningProtocol <: MentalProtocol end

function plan! end


"""
    PlanningModule(::T, ...)::MentalModule{T} where {T<:PlanningProtocol}

Constructor that should be implemented by each `PlanningProtocol`
"""
function PlanningModule end

@with_kw struct CollisionCounter <: PlanningProtocol
    "Target appearance"
    mat::Material
    "Distance tolerance"
    tol::Float64 = 1E-4
end

mutable struct CollisionState <: MentalState{CollisionCounter}
    expectation::Float64
end

function PlanningModule(p::CollisionCounter)
    MentalModule(p, CollisionState(0.0))
end

function plan!(planner::MentalModule{T},
               attention::MentalModule{A},
               perception::MentalModule{V}
    ) where {T<:CollisionCounter,
             A<:AttentionProtocol,
             V<:PerceptionProtocol}

    protocol, state = parse(planner)

    w = estimate_marginal(perception,
                          plan_with_delta_pi!,
                          (protocol, attention))

    state.expectation = w

    return nothing
end

function plan_with_delta_pi!(pl::CollisionCounter, att::MentalModule{A}, tr::InertiaTrace
                             ) where {A<:AttentionProtocol}
    _, wm = get_args(tr)
    states = get_retval(tr)
    state = last(states)
    @unpack walls = wm

    @unpack singles, ensembles = state
    ns = length(singles)
    ne = length(ensembles)
    ep = 0.0
    @inbounds for j = 1:ns
        dpi = 0.0
        single = singles[j]
        # ignore other color objects
        single.mat == pl.mat || continue
        # consider each wall
        # REVIEW: maybe just look at closest wall?
        for k = 1:4 # each wall
            (_ep, _dpi) = colprob_and_agrad(single, walls[k])
            ep += _ep
            dpi += _dpi
        end
        update_dPi!(att, single, log(dpi))
    end
    # TODO: energy for ensembles
    #
    return ep
end


function colprob_and_agrad(x::InertiaSingle, w::Wall)
    pos = get_pos(x)
    d = w.d - sum(w.normal .* pos) # - s.size
    p = sigmoid(d, x.size) # x, x0, m
    dpdx = sigmoid_grad(d, x.size)
    (p, dpdx)
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
