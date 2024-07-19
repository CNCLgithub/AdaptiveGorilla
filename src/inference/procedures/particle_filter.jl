export AdaptiveParticleFilter,
    APChain

using GenParticleFilters
using Gen_Compose
using Gen_Compose: initial_args, initial_constraints,
    AuxillaryState, PFChain

@with_kw struct AdaptiveParticleFilter <: Gen_Compose.AbstractParticleFilter
    particles::Int = 1
    ess::Real = particles * 0.5
    attention::AttentionProtocol
end

function Gen_Compose.PFChain{Q, P}(q::Q,
                                   p::P,
                                   n::Int,
                                   i::Int = 1) where
    {Q<:SequentialQuery,
     P<:AdaptiveParticleFilter}

    state = Gen_Compose.initialize_procedure(p, q)
    aux = AuxState(p.attention)
    return PFChain{Q, P}(q, p, state, aux, i, n)
end

APChain = PFChain{SequentialQuery, AdaptiveParticleFilter}

function Gen_Compose.step!(chain::PFChain{<:SequentialQuery, <:AdaptiveParticleFilter})
    @unpack query, proc, state, step = chain
    @show step
    squery = query[step]
    @unpack args, argdiffs, observations = squery
    # Resample before moving on...
    if effective_sample_size(state) < proc.ess
        # Perform residual resampling, pruning low-weight particles
        println("Pruning particles...")
        pf_residual_resample!(state)
    end
    # update the state of the particles
    Gen.particle_filter_step!(state, args, argdiffs,
                              observations)
    @unpack attention = proc
    apply_protocol!(chain, attention)
    return nothing
end
