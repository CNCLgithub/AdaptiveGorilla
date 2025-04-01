export HyperFilter,
       HyperState,
       PerceptionModule

@with_kw struct HyperFilter{P<:AbstractParticleFilter
                  } <: PerceptionProtocol
    "Number of hyper particle chains"
    h::Int = 4
    "Number of observations per epoch"
    dt::Int = 12
    "Particle filter procedure"
    pf::P
    "Maps an integer to the time address in GM"
    time_prefix::Function = x::Int -> :kernel => x => :xs
    "Temperate for hyper particle resampling"
    resample_temp::Float64 = 1.0
end

mutable struct HyperState <: MentalState{HyperFilter}
    chains::Vector{APChain}
    "age of local chains"
    age::Int64
end

function HyperState(m::HyperFilter, q::IncrementalQuery)
    pf = m.pf
    chains = Vector{APChain}(undef, m.h)
    @inbounds for i = 1:m.h
        chains[i] = Gen_Compose.initialize_chain(pf, q, m.dt)
    end
    return HyperState(chains, 1)
end

function PerceptionModule(m::HyperFilter, q::IncrementalQuery)
    s = HyperState(m, q)
    MentalModule(m, s)
end

function perceive!(perception::MentalModule{T},
                   attention::MentalModule{A},
                   obs::Gen.ChoiceMap
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
        # Stage 2: attention, records dS
        attend!(chain, attention)
        chain.step += 1
        # update reference in perception module
        x.chains[i] = chain
    end
    x.age += 1

    return nothing
end

function estimate_marginal(perception::MentalModule{T},
                           func::Function,
                           args::Tuple
    ) where {T<:HyperFilter}

    pf, st = parse(perception)

    m = 0.0

    for i = 1:pf.h
        m += estimate_marginal(st.chains[i], func, args)
    end

    m *= 1.0 / pf.h

    return m
end
