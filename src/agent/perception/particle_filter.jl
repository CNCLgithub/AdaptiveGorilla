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
    PFChain{Q, P}(q, p, state, aux, i, n)
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
    ws = state.log_weights
    mass = logsumexp(ws)
    acc = -Inf
    @inbounds for i = 1:length(ws)
        # TODO: Investigate and remove
        # if sum(ws) != 1.0 || isnan(ws[i]) || isinf(ws[i])
        #     error("Issue with particle weights: $(ws)")
        # end
        v = func(args..., state.traces[i])
        w = ws[i] - mass
        acc = logsumexp(acc, w + v)
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
