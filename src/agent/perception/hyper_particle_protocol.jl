export HyperFilter,
       HyperState

struct HyperFilter{P<:AbstractParticleFilter
                  } <: PerceptionModule{W}
    "Number of hyper particle chains"
    h::Int
    "Number of observations per epoch"
    dt::Int
    "Particle filter procedure"
    pf::P
    # e.g., :kernel => x.time => :xs
    time_prefix::Function
end

mutable struct HyperState <: MentalState{HyperFilter}
    aux::AuxState
    chains::Vector{APChain}
    "age of local chains"
    age::Int64
end

function HyperState(m::HyperFilter, gm::GenarativeFunction,
        wm::W, ws::WorldState{<:W}) where {W<:WorldModel}
    pf = m.pf
    aux = AuxState(pf.attention)
    chains = Vector{APChain}(undef, m.h)
    @inbounds for i = 1:m.h
        chains[i] = init_chain_common_aux(gm, wm, ws, m.pf)
    end
    return HyperState(aux, chains)
end

function PerceptionModule(m::HyperFilter, gm::GenarativeFunction,
        wm::W, ws::WorldState{<:W}) where {W<:WorldModel}
    s = HyperState(m, gm, wm, ws)
    MentalModule(m, s)
end

function perceive!(perception<:MentalModule{T},
                   attention<:MentalModule{A},
                   update::Gen.ChoiceMap
    ) where {T<:HyperFilter, A<:AttentionProtocol}

    pm, x = parse(perception)
    new_args = (x.age,)
    cm = choicemap()
    set_submap!(cm, pm.time_prefix(x.age), obs)

    # update
    for i = 1:pm.h
        chain = x.chains[i]
        # add new observation
        chain.query = increment(chain.query, cm, new_args)
        # Stage 1: initial approximation of S^t
        step!(chain)
        # Stage 2: attention
        attend!(chain, attention)
        chain.step += 1
        # update reference in perception module
        x.chains[i] = chain
    end
    x.age += 1

    return nothing
end

function estimate_marginal(perception<:MentalModule{T},
                           func::Function,
                           args::Tuple
    ) where {T<:HyperFilter}

    pf, st = parse(perception)

    m = 0.0

    for i = 1:pm.h
        m += estimate_marginal(st.chains[i], func, args)
    end

    m *= 1.0 / pm.h

    return m
end

function resample_chains!(perception<:MentalModule{T},
                          ws::Vector{Float64}) where {T<:HyperFilter}
    pm, x = parse(perception)



end
