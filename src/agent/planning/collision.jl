export CollisionCounter,
    CollisionState

"""
    ($TYPEDEF)

Protocol for counting collisions of objects that match target appearance `mat`.

---

$(TYPEDFIELDS)

"""
@with_kw struct CollisionCounter <: PlanningProtocol
    "Target appearance"
    mat::Material
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
function module_step!(planner::MentalModule{T},
                      t::Int,
                      attention::MentalModule{A},
                      perception::MentalModule{V}
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
            # println("LOG COL PROB: $(w)")
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

function closest_wall(object::InertiaObject, walls)
    wall = walls[1]
    x = get_pos(object)
    v = get_vel(object)
    distance = abs(wall.d - dot(x, wall.normal))
    wall_idx = 1
    for i = 2:4
        wall = walls[i]
        x = get_pos(object)
        v = get_vel(object)
        d = abs(wall.d - dot(x, wall.normal))
        if distance > d
            distance = d
            wall_idx = i
        end
    end
    return wall_idx
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
            closest = walls[closest_wall(single, walls)]
            (_colprob, _dpi) = colprob_and_agrad(single, closest)
            colprob = logsumexp(colprob, _colprob)
            dpi = logsumexp(dpi, _dpi)
            # println("pos: $(get_pos(single)) \n vel: $(get_vel(single))")
            # @show closest
            # @show _colprob
            # @show colprob
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
            closest = walls[closest_wall(x, walls)]
            (_colprob, _dpi) = colprob_and_agrad(x, closest)
            colprob = logsumexp(colprob, _colprob)
            dpi = logsumexp(dpi, _dpi)
            # println("pos: $(get_pos(x)) \n vel: $(get_vel(x)) \n spread: $(get_var(x))")
            # @show closest
            # @show _colprob
            # @show colprob
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

function colprob_and_agrad(obj::InertiaSingle, w::Wall, radius = 5.0)
    # Distance between object and wall
    x = get_pos(obj)
    v = get_vel(obj)
    distance = abs(w.d - dot(x, w.normal)) - radius

    v_orth = min(dot(v, w.normal), 4.5)
    # Average time (steps) to collision
    dt = v_orth < 1E-5 ? 100.0 : distance / v_orth
    
    # Penalty for higher angular velocity
    sigma = 1.0 / exp(-abs(get_avel(obj)))

    # Z score of 1 step in the future
    z = (1.0 - dt) / sigma
    # CCDF up to 1 step
    lcdf = Distributions.logcdf(standard_normal, z)
    # pdf is the derivative of the cdf
    dpdz = Distributions.logpdf(standard_normal, z)
    # @show x
    # @show v
    # @show v_orth
    # @show get_avel(obj)
    # @show dt
    # @show sigma
    # @show z
    # @show lcdf
    # @show dpdz
    (lcdf, dpdz)
end

function colprob_and_agrad(obj::InertiaEnsemble, w::Wall)
    r = rate(obj)
    prop_light = materials(obj)[1]
    isapprox(prop_light, 0; atol=1e-4) && return (-Inf, -Inf)
    lpl = log(prop_light)
    x = get_pos(obj)
    v = get_vel(obj)
    distance = abs(w.d - dot(x, w.normal))

    # Average time for the ensemble to reach
    # the wall
    v_orth = dot(v, w.normal)
    dt = v_orth < 1E-5 ? 200.0 : distance / v_orth

    # Variance increases with speed as before,
    # but decreases with ensemble.
    # This is because ensemble spread relates
    # to its entropy, with more entropy
    # increasing the variance over velocity direction
    sigma = 5.0  / sqrt(get_var(obj))
    z = (1.0 - dt) / sigma
    # CDF up to 1 step
    pcol = Distributions.logcdf(standard_normal, z)
    # Scale by the number of objects,
    # and the proportion that are light
    lcdf = r * pcol + lpl

    # pdf is the derivative of the cdf
    dpdz = (r-1) * Distributions.logpdf(standard_normal, z) + lpl + r
    (lcdf, dpdz)
end

# VISUALS

using MOTCore: _draw_text

function render_frame(x::MentalModule{P}, t::Int) where{P<:CollisionCounter}
    protocol, state = mparse(x)
    c = round(state.expectation; digits = 2)
    _draw_text("Bounce weight: $(c)", [-380, 380.])
end
