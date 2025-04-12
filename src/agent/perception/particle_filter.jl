export AdaptiveParticleFilter,
    APChain

using GenParticleFilters
using Gen_Compose: initial_args, initial_constraints,
    AuxillaryState, PFChain

@with_kw struct AdaptiveParticleFilter <: Gen_Compose.AbstractParticleFilter
    particles::Int = 1
    ess::Real = particles * 0.5
end


const APChain = PFChain{<:IncrementalQuery, AdaptiveParticleFilter}

function PFChain{Q, P}(q::Q,
                       p::P,
                       n::Int,
                       i::Int = 1) where
        {Q<:IncrementalQuery,  P<:AdaptiveParticleFilter}
    state = Gen.initialize_particle_filter(q.model,
                                           q.args,
                                           q.constraints,
                                           p.particles)
    aux = EmptyAuxState()
    PFChain{IncrementalQuery, AdaptiveParticleFilter}(q, p, state, aux, i, n)
end

function Gen_Compose.step!(chain::PFChain{<:IncrementalQuery, <:AdaptiveParticleFilter})
    @unpack query, proc, state, step = chain
    @unpack args, argdiffs, constraints = query
    # Resample before moving on...
    if effective_sample_size(state) < proc.ess
        # Perform residual resampling, pruning low-weight particles
        pf_residual_resample!(state)
    end
    # update the state of the particles
    Gen.particle_filter_step!(state, args, argdiffs,
                              query.constraints)
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

"""
Returns the MAP trace
"""
function retrieve_map(chain::APChain)
    @unpack state = chain
    state.traces[argmax(state.log_weights)]
end

function reinit_chain(chain::APChain, template::InertiaTrace,
                      cm = choicemap())
    pf = estimator(chain)
    q = estimand(chain)
    steps = chain.steps # dt
    _, wm, _ = q.args
    ws = get_last_state(template)
    args = (0, wm, ws)
    argdiffs = (Gen.UnknownChange(), Gen.NoChange(), Gen.NoChange())
    q = IncrementalQuery(q.model, cm, args, argdiffs, 1)
    Gen_Compose.initialize_chain(pf, q, steps)
end
