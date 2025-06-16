export CollisionCounter,
    CollisionState

"""
(TYPEDEF)

Protocol for counting collisions of objects that match target appearance `mat`.

---

$(TYPEDFIELDS)

"""
@with_kw struct CollisionCounter <: PlanningProtocol
    "Target appearance"
    mat::Material
    "Distance tolerance"
    tol::Float64 = 1E-4
    "Tick rate"
    tick_rate::Int = 1
end

mutable struct CollisionState <: MentalState{CollisionCounter}
    "Count"
    expectation::Float64
    "Amount of frames until next estimate"
    cooldown::Int64
end

function PlanningModule(p::CollisionCounter)
    MentalModule(p, CollisionState(0.0, 0))
end

# helper to extract planning state
function planner_expectation(pm::MentalModule{T}) where {T<:CollisionCounter}
    planner, state = mparse(pm)
    state.expectation
end

"""

$(SIGNATURES)

Computes the marginal over collision counts.

Also updates the \$\\delta \\pi\$ records in the attention module.
"""
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
        if rand() < w
            state.expectation += 1
        end
        # state.expectation += w
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
    ep = -Inf
    @inbounds for j = 1:ns
        dpi = -Inf
        single = singles[j]
        # only consider targets
        if single.mat == pl.mat
            # REVIEW: maybe just look at closest wall?
            for k = 1:4 # each wall
                (_ep, _dpi) = colprob_and_agrad(single, walls[k])
                ep = logsumexp(ep, _ep)
                dpi = logsumexp(dpi, _dpi)
            end
        end
        update_dPi!(att, single, dpi)
    end
    @inbounds for j = 1:ne
        dpi = -Inf
        x = ensembles[j]
        # target proportion of ensemble
        w = x.matws[Int64(pl.mat)]
        if w  > 0.1
            for k = 1:4 # each wall
                (_ep, _dpi) = colprob_and_agrad(x, walls[k])
                ep = logsumexp(ep, _ep)
                dpi = logsumexp(dpi, _dpi)
            end
        end
        update_dPi!(att, x, dpi)
    end
    return exp(ep)
end

function colprob_and_agrad(pos::S2V, w::Wall)
    # HACK: assumes object radius of 10
    d = max(0.1, (w.d - sum(w.normal .* pos)) - 10)
    p = exp(min(0.0, -log(d) - 1))
    dpdx = min(1.0, 1 / d )
    (p, dpdx)
end

function colprob_and_agrad(obj::InertiaSingle, w::Wall)
    x = get_pos(obj)
    v = get_vel(obj)
    v_orth = dot(v, w.normal)
    distance = abs(w.d - dot(x, w.normal))
    # Distribution over near future
    var = 0.5 * abs(v_orth)
    mu = 0.5 * v_orth + get_size(obj)
    pred = Normal(mu, var)
    # CCDF up to wall
    p = Distributions.logccdf(pred, distance)
    # pdf is the derivative of the cdf
    dpdx = Distributions.logpdf(pred, distance)
    (p, dpdx)
end

function colprob_and_agrad(obj::InertiaEnsemble, w::Wall)
    r = rate(obj)
    l = 1.0 - materials(obj)[1] # prop not light
    penalty = iszero(l) ? 0.0 : log(r) + log(l)
    x = get_pos(obj)
    vel = get_vel(obj)
    var = get_var(obj)
    vel_orth = dot(vel, w.normal)
    distance = abs(w.d - dot(x, w.normal))
    # Distribution over near future
    # Variance integrates ensemble spread
    # This dilutes probability density
    sigma = 0.5 * (abs(vel_orth) + 0.1*var)
    mu = 0.5 * vel_orth
    pred = Normal(mu, sigma)
    # display(pred)
    # @show distance
    # CCDF up to wall
    p = Distributions.logccdf(pred, distance)
    p -= penalty
    # pdf is the derivative of the cdf
    dpdx = Distributions.logpdf(pred, distance)
    dpdx -= penalty
    (p, dpdx)
end

# VISUALS

using MOTCore: _draw_text

function render_frame(x::MentalModule{P}, t::Int) where{P<:CollisionCounter}
    protocol, state = mparse(x)
    c = round(state.expectation; digits = 2)
    _draw_text("Bounce weight: $(c)", [-380, 380.])
end
