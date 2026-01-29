export HyperFilter,
    HyperState,
    PerceptionModule,
    estimate_marginal

@with_kw struct HyperFilter{P<:AbstractParticleFilter
                  } <: PerceptionProtocol
    "Number of hyper particle chains"
    h::Int = 4
    "Number of observations per epoch"
    dt::Int = 12
    "Particle filter procedure"
    pf::P
    "Maps an integer to the time address in GM"
    time_prefix::Function = _kernel_prefix
    "Temperate for hyper particle resampling"
    resample_temp::Float64 = 1.0
end

mutable struct HyperState <: MentalState{HyperFilter}
    chains::Vector{APChain}
    new_chains::Vector{APChain}
    "age of local chains"
    age::Int64
end

function HyperState(m::HyperFilter, q::IncrementalQuery)
    pf = m.pf
    chains = Vector{APChain}(undef, m.h)
    new_chains = Vector{APChain}(undef, m.h)
    @inbounds for i = 1:m.h
        chains[i] = Gen_Compose.initialize_chain(pf, q, m.dt)
    end
    return HyperState(chains, new_chains, 1)
end

function PerceptionModule(m::HyperFilter, q::IncrementalQuery)
    s = HyperState(m, q)
    MentalModule(m, s)
end

function module_step!(perception::MentalModule{T},
                      t::Int,
                      obs::Gen.ChoiceMap
    ) where {T<:HyperFilter}

    pm, x = mparse(perception)
    new_args = (x.age,)
    cm = choicemap()
    n = 1
    while true
        if !has_value(obs, n)
            break
        end
        cm[pm.time_prefix(x.age, n)] = obs[n]
        n += 1
    end

    # update
    for i = 1:pm.h
        chain = x.chains[i]
        # add new observation
        chain.query = increment(chain.query, cm, new_args)
        # Stage 1: initial approximation of S^t
        step!(chain)
        chain.step += 1
        # update reference in perception module
        x.chains[i] = chain
    end
    x.age += 1

    return nothing
end

function reset_state!(visstate::HyperState, visp::HyperFilter)
    visstate.age = 1
    temp_chains = visstate.chains
    visstate.chains = visstate.new_chains
    visstate.new_chains = temp_chains
    return nothing
end

function estimate_marginal(perception::MentalModule{T},
                           func::Function,
                           args::Tuple
    ) where {T<:HyperFilter}
    pf, st = mparse(perception)
    m = -Inf
    for i = 1:pf.h
        v = estimate_marginal(st.chains[i], func, args)
        m = logsumexp(m, v)
    end
    m -= log(pf.h)
    return m
end
