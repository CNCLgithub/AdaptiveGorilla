export Planner,
    PlanningModule,
    plan!,
    CollisionCounter

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
    "Tick rate"
    tick_rate::Int = 1
end

mutable struct CollisionState <: MentalState{CollisionCounter}
    expectation::Float64
end

function PlanningModule(p::CollisionCounter)
    MentalModule(p, CollisionState(0.0))
end

function plan!(planner::MentalModule{T},
               attention::MentalModule{A},
               perception::MentalModule{V},
               t::Int
    ) where {T<:CollisionCounter,
             A<:AttentionProtocol,
             V<:PerceptionProtocol}

    protocol, state = mparse(planner)
    if t > 0 && t % protocol.tick_rate == 0
        w = estimate_marginal(perception,
                              plan_with_delta_pi!,
                              (protocol, attention))
        state.expectation += w
    end
    return nothing
end

function plan_with_delta_pi!(
    pl::CollisionCounter, att::MentalModule{A}, tr::InertiaTrace
    ) where {A<:AttentionProtocol}
    _, wm = get_args(tr)
    @unpack walls = wm
    state = get_last_state(tr)
    @unpack singles, ensembles = state
    ns = length(singles)
    ne = length(ensembles)
    ecount = 0.0
    ep = -Inf
    @inbounds for j = 1:ns
        dpi = -Inf
        single = singles[j]
        # only consider targets
        single.mat != pl.mat && continue
        # consider each wall
        # REVIEW: maybe just look at closest wall?
        for k = 1:4 # each wall
            (_ep, _dpi) = colprob_and_agrad(single, walls[k])
            ep = logsumexp(ep, _ep)
            dpi = logsumexp(dpi, _dpi)
            if log(rand()) < ep
                ecount += 1
                ep = -Inf
            end
        end
    end
    update_dPi!(att, single, dpi)
    @inbounds for j = 1:ne
        dpi = -Inf
        x = ensembles[j]
        # target proportion of ensemble
        w = x.matws[Int64(pl.mat)]
        for k = 1:4 # each wall
            (_ep, _dpi) = colprob_and_agrad(x, walls[k])
            ep = logsumexp(ep, log(w) + _ep)
            dpi = logsumexp(dpi, log(w) + _dpi)
            if log(rand()) < ep
                ecount += 1
                ep = -Inf
            end
        end
        update_dPi!(att, x, dpi)
    end
    return ecount
end

function colprob_and_agrad(pos::S2V, w::Wall)
    # HACK: assumes object radius of 10
    d = max(0.1, (w.d - sum(w.normal .* pos)) - 10)
    p = exp(min(0.0, -log(d) - 1))
    dpdx = min(1.0, 1 / d )
    # p = fast_sigmoid(z) # x, x0, m
    # dpdx = abs(fast_sigmoid_grad(z))
    (p, dpdx)
end

function colprob_and_agrad(obj::InertiaSingle, w::Wall)
    x = get_pos(obj)
    v = get_vel(obj)
    v_orth = dot(v, w.normal)
    distance = abs(w.d - dot(x, w.normal))
    # Distribution over near future
    var = 0.5 * v_orth
    mu = 0.5 * v_orth + get_size(obj)
    pred = Normal(mu, var)
    # CCDF up to wall
    p = logccdf(pred, distance)
    # pdf is the derivative of the cdf
    dpdx = logpdf(pred, distance)
    (p, dpdx)
end

function colprob_and_agrad(obj::InertiaEnsemble, w::Wall)
    x = get_pos(obj)
    vel = get_vel(obj)
    var = get_var(obj)
    vel_orth = dot(vel, w.normal)
    var_orth = dot(var, w.normal)
    distance = abs(w.d - dot(x, w.normal))
    # Distribution over near future
    # Variance integrates ensemble spread
    # This dilutes probability density
    var = 0.5 * (v_orth + var_orth)
    mu = 0.5 * v_orth
    pred = Normal(mu, var)
    # CCDF up to wall
    p = logccdf(pred, distance)
    # pdf is the derivative of the cdf
    dpdx = logpdf(pred, distance)
    (p, dpdx)
end

# VISUALS

using MOTCore: _draw_text

function render_frame(x::MentalModule{P}, t::Int) where{P<:CollisionCounter}
    protocol, state = mparse(x)
    c = round(state.expectation; digits = 2)
    _draw_text("Bounce weight: $(c)", [-380, 380.])
end
