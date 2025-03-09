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


const APChain = PFChain{<:IncrementalQuery, AdaptiveParticleFilter}

function init_chain_common_aux(
        model::Gen.GenerativeFunction,
        wm::T, init_state::WorldState{T},
        proc::AdaptiveParticleFilter,
        aux::AuxillaryState,
    ) where {T<:WorldModel}
    args = (0, init_state, wm) # t = 0
    # argdiffs: only `t` changes
    argdiffs = (Gen.UnknownChange(), Gen.NoChange(), Gen.NoChange())
    query = IncrementalQuery(model, Gen.choicemap(),
                             args, argdiffs, 1)
    APChain(query, proc, state, aux, 1, typemax(1))
end


function Gen_Compose.step!(chain::PFChain{<:IncrementalQuery, <:AdaptiveParticleFilter})
    @unpack query, proc, state, step = chain
    @unpack args, argdiffs, constraints = query
    # Resample before moving on...
    if effective_sample_size(state) < proc.ess
        # Perform residual resampling, pruning low-weight particles
        println("Pruning particles...")
        pf_residual_resample!(state)
    end
    # update the state of the particles
    Gen.particle_filter_step!(state, args, argdiffs,
                              observations)
    return nothing
end

function estimate_marginal(chain::PFChain{<:IncrementalQuery, <:AdaptiveParticleFilter},
                           func::Function,
                           args::Tuple)

    @unpack state = chain
    ws = get_norm_weights(state)
    acc = 0.0
    @inbounds for i = 1:length(ws)
        acc += ws[i] * func(args..., state.traces[i])
    end
    return acc
end
