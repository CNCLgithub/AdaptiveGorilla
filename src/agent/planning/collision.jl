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
    "Threshold to increment collision"
    threshold::Float64 = 20.0
end

mutable struct CollisionState <: MentalState{CollisionCounter}
    "Count"
    expectation::Float64
    "Amount of frames until next estimate"
    cooldown::Int64
    "Previous collision location"
    prev_spot::S2V
end

function PlanningModule(p::CollisionCounter)
    MentalModule(p, CollisionState(0.0, 0, S2V(0., 0.)))
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
        # updates dPi internally, see `plan_with_delta_pi!`
        map_colprob, map_loc =
            estimate_marginal_outer(protocol, attention, perception)

        # Only consider new collision if either:
        # - it has been sufficient time
        # - it is in a new spot
        
        # println("TIME $(t) [-t: $(state.cooldown)], COL PROB: $(map_colprob), D: $(d)")
        # @show map_loc
        # @show state.prev_spot
        if state.cooldown == 0
            d = norm(state.prev_spot - map_loc)
            if log(rand()) < map_colprob && d > protocol.threshold 
                state.expectation += 1
                state.cooldown = protocol.cooldown
                state.prev_spot = map_loc
                # println("COUNT: $(state.expectation)")
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
    cum_colprob = -Inf
    map_colprob = -Inf
    map_loc = S2V(0, 0)
    @inbounds for j = 1:ns
        dpi = -Inf
        single = singles[j]
        # only consider targets
        if single.mat == pl.mat
            closest = walls[closest_wall(single, walls)]
            (_colprob, _dpi) = colprob_and_agrad(single, closest)
            # update record of most likely col
            if _colprob > map_colprob
                map_colprob = _colprob
                map_loc = get_pos(single)
            end
            cum_colprob = logsumexp(cum_colprob, _colprob)
            dpi = logsumexp(dpi, _dpi)
        end
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
            # update record of most likely col
            if _colprob > map_colprob
                map_colprob = _colprob
                sampled_pos = broadcasted_normal(
                    get_pos(x),
                    get_var(x)
                )
                map_loc = S2V(sampled_pos)
            end
            cum_colprob = logsumexp(cum_colprob, _colprob)
            dpi = logsumexp(dpi, _dpi)
        end
        update_dPi!(att, x, dpi)
    end
    return (cum_colprob, map_colprob, map_loc)
end

function colprob_and_agrad(obj::InertiaSingle, w::Wall, radius = 5.0)
    # Distance between object and wall
    x = get_pos(obj)
    v = get_vel(obj)
    distance = abs(w.d - dot(x, w.normal)) - radius
    # Average time (steps) to collision
    v_orth = dot(v, w.normal)
    dt = v_orth < 1E-5 ? 100.0 : distance / v_orth
    # Penalty for higher angular velocity
    sigma = 0.5 * exp(0.5*abs(get_avel(obj)))
    # Z score of 1 step in the future
    z = (1.0 - dt) / sigma
    # CCDF up to 1 step
    lcdf = Distributions.logcdf(standard_normal, z)
    # pdf is the derivative of the cdf
    dpdz = Distributions.logpdf(standard_normal, z)
    # Uncomment to verify high-col prob
    # if lcdf > -0.5
    #     @show x
    #     @show v
    #     @show v_orth
    #     @show get_avel(obj)
    #     @show distance
    #     @show dt
    #     @show sigma
    #     @show z
    #     @show lcdf
    #     @show dpdz
    # end
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
    dt = v_orth < 1E-5 ? 100.0 : distance / v_orth

    # Variance increases with speed as before,
    # but decreases with ensemble.
    # This is because ensemble spread relates
    # to its entropy, with more entropy
    # increasing the variance over velocity direction
    sigma = 1.0 / get_var(obj)
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


#################################################################################
# Marginal operations over perception                                           #
#################################################################################

function estimate_marginal_outer(
    pl::CollisionCounter,
    att::MentalModule{A},
    perception::MentalModule{T}
    )::Tuple{Float64, S2V} where {T<:HyperFilter, A<:AttentionProtocol}

    pf, st = mparse(perception)
    map_colprob = -Inf
    map_loc = S2V(0, 0)
    for i = 1:pf.h
        _map_colprob, _map_loc =
            estimate_marginal_inner(st.chains[i], pl, att)
        if _map_colprob > map_colprob
            map_colprob = _map_colprob
            map_loc = _map_loc
        end
    end
    return map_colprob, map_loc
end

function estimate_marginal_inner(
    chain::PFChain{<:IncrementalQuery, <:AdaptiveParticleFilter},
    pl::CollisionCounter, att::MentalModule{A}
    ) where {A<:AttentionProtocol}
    @unpack state = chain
    ws = state.log_weights
    mass = logsumexp(ws)

    map_colprob = -Inf
    map_loc = S2V(0, 0)
    @inbounds for i = 1:length(ws)
        (_, _map_colprob, _map_loc) =
            plan_with_delta_pi!(pl, att, state.traces[i])

        _map_colprob += ws[i] - mass
        if _map_colprob > map_colprob
            map_colprob = _map_colprob
            map_loc = _map_loc
        end
    end
    return (map_colprob, map_loc)
end
