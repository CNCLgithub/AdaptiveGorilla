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
    ep = 0.0
    @inbounds for j = 1:ns
        dpi = 0.0
        single = singles[j]
        # only consider targets
        if single.mat == pl.mat
            # consider each wall
            # REVIEW: maybe just look at closest wall?
            for k = 1:4 # each wall
                (_ep, _dpi) = colprob_and_agrad(single, walls[k])
                ep += _ep
                dpi += _dpi
            end
        end
        update_dPi!(att, single, log(dpi))
    end
    @inbounds for j = 1:ne
        dpi = 0.0
        x = ensembles[j]
        # # target proportion of ensemble
        # w = x.matws[Int64(pl.mat)]
        # for k = 1:4 # each wall
        #     (_ep, _dpi) = colprob_and_agrad(x, walls[k])
        #     ep += _ep
        #     dpi += _dpi
        # end
        # ep *= w
        # dpi *= w
        update_dPi!(att, x, log(dpi))
    end
    return ep
end


function colprob_and_agrad(x::InertiaSingle, w::Wall)
    pos = get_pos(x)
    d = (w.d - sum(w.normal .* pos))
    z = d / x.size
    p = fast_sigmoid(z) # x, x0, m
    dpdx = abs(fast_sigmoid_grad(z))
    (p, dpdx)
end

function colprob_and_agrad(x::InertiaEnsemble, w::Wall)
    pos = get_pos(x)
    d = (w.d - sum(w.normal .* pos))
    z = d / (sqrt(x.var) * x.rate)
    p = fast_sigmoid(z) # x, x0, m
    dpdx = abs(fast_sigmoid_grad(z))
    (p, dpdx)
end

using MOTCore: _draw_text

function render_frame(x::MentalModule{P}, t::Int) where{P<:CollisionCounter}
    protocol, state = mparse(x)
    c = round(state.expectation / t; digits = 2)
    _draw_text("Bounce weight: $(c)", [-380, 380.])
end
