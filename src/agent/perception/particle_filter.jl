export AdaptiveParticleFilter,
    APChain

using GenParticleFilters
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

function PFChain{Q, P}(q::Q,
                       p::P,
                       n::Int,
                       i::Int = 1) where
        {Q<:IncrementalQuery,  P<:AdaptiveParticleFilter}
    state = initialize_procedure(p, q)
    aux = EmptyAuxState()
    PFChain{Q, P}(q, p, state, aux, i, length(q))
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

"""
Returns the MAP trace
"""
function retrieve_map(chain::APChain)
    @unpack state = chain
    state.traces[argmax(state.log_weigths)]
end

function reinit_chain(chain::APChain)
    traces = Vector{Any}(undef, num_particles)
    log_weights = Vector{Float64}(undef, num_particles)
    for i=1:num_particles
        (traces[i], log_weights[i]) = generate(model, model_args, observations)
    end
    ParticleFilterState{U}(traces, Vector{U}(undef, num_particles),
        log_weights, 0., collect(1:num_particles))
end
