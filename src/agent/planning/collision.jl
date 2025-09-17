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
    "Counting cool down"
    cooldown::Int = 5
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
    if (t > 0 && t % protocol.tick_rate == 0)
        # updates dPi
        w = estimate_marginal(perception,
                              plan_with_delta_pi!,
                              (protocol, attention))
        if state.cooldown == 0
            if log(rand()) < w
                state.expectation += 1
                state.cooldown = protocol.cooldown
            end
        else
            state.cooldown -= protocol.tick_rate
        end
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
    colprob = -Inf
    @inbounds for j = 1:ns
        dpi = -Inf
        single = singles[j]
        # only consider targets
        if single.mat == pl.mat
            # REVIEW: maybe just look at closest wall?
            for k = 1:4 # each wall
                (_colprob, _dpi) = colprob_and_agrad(single, walls[k])
                colprob = logsumexp(colprob, _colprob)
                dpi = logsumexp(dpi, _dpi)
            end
        end
        # @show dpi
        update_dPi!(att, single, dpi)
    end
    @inbounds for j = 1:ne
        dpi = -Inf
        x = ensembles[j]
        # target proportion of ensemble
        w = x.matws[Int64(pl.mat)]
        if w  > 0.1
            for k = 1:4 # each wall
                (_colprob, _dpi) = colprob_and_agrad(x, walls[k])
                colprob = logsumexp(colprob, _colprob)
                dpi = logsumexp(dpi, _dpi)
            end
        end
        update_dPi!(att, x, dpi)
    end
    return colprob
end

# HACK: assumes object radius
function colprob_and_agrad(pos::S2V, w::Wall, radius::Float64 = 5)
    p = exp(min(0.0, -log(d) - 1))
    dpdx = min(1.0, 1 / d )
    (p, dpdx)

    distance = max(0.1, (w.d - sum(w.normal .* pos)) - 10)

    distance = abs(w.d - dot(x, w.normal))

    # Distance distribution over near future
    v_orth = dot(v, w.normal)
    mu = 0.5 * v_orth + get_size(obj)
    sigma = 5.0 * abs(v_orth)
    z = (distance - mu) / sigma
    # CCDF up to wall
    lcdf = Distributions.logcdf(standard_normal, z)

    # Account for heading - low prob if object is facing away
    log_angle = log(0.5 * (dot(normalize(v), w.normal) + 1.0))
    log_angle = clamp(log_angle, -10.0, 0.0)
    logcolprob = log_angle + log1mexp(lcdf) # Pr(col) = 1 - Pr(!col)

    # pdf is the derivative of the cdf
    dpdz = log_angle + log_grad_normal_cdf_erfcx(z)
    (logcolprob, dpdz)
end


function colprob_and_agrad(obj::InertiaSingle, w::Wall)
    # Distance between object and wall
    x = get_pos(obj)
    v = get_vel(obj)
    distance = abs(w.d - dot(x, w.normal))

    # Distance distribution over near future
    v_orth = dot(v, w.normal)
    mu = 0.5 * v_orth + get_size(obj)
    sigma = 5.0 * abs(v_orth)
    z = (distance - mu) / sigma
    # CCDF up to wall
    lcdf = Distributions.logcdf(standard_normal, z)

    # Account for heading - low prob if object is facing away
    log_angle = log(0.5 * (dot(normalize(v), w.normal) + 1.0))
    log_angle = clamp(log_angle, -10.0, 0.0)
    logcolprob = log_angle + log1mexp(lcdf) # Pr(col) = 1 - Pr(!col)

    # pdf is the derivative of the cdf
    dpdz = log_angle + log_grad_normal_cdf_erfcx(z)
    (logcolprob, dpdz)
end

# function colprob_and_agrad(obj::InertiaEnsemble, w::Wall)
#     # Ensemble representations do not maintain persistent
#     # object trajectories, so they cannot inform
#     # collision counting
#     (-Inf, -Inf)
# end
function colprob_and_agrad(obj::InertiaEnsemble, w::Wall)
    r = rate(obj)
    prop_light = materials(obj)[1]
    isapprox(prop_light, 0; atol=1e-4) && return (-Inf, -Inf)
    x = get_pos(obj)
    sigma = sqrt(get_var(obj))
    distance = abs(w.d - dot(x, w.normal))
    # Variance integrates ensemble spread
    # This dilutes probability density
    z = distance / sigma
    # probability of a single object not colliding
    lcdf = Distributions.logcdf(standard_normal, z)
    # probability of any collision * prop targets
    p = log1mexp(r * lcdf)
    dpdx = -log(prop_light) * log_grad_normal_cdf_erfcx(z)
    (p, dpdx)
end

# VISUALS

using MOTCore: _draw_text

function render_frame(x::MentalModule{P}, t::Int) where{P<:CollisionCounter}
    protocol, state = mparse(x)
    c = round(state.expectation; digits = 2)
    _draw_text("Bounce weight: $(c)", [-380, 380.])
end
