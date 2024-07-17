export UniformProtocol

@with_kw struct UniformProtocol <: AttentionProtocol
    """Moves over singles"""
    single_block! = ReweightBlock(single_ancestral_proposal, 10)
    """Moves over ensemble"""
    ensemble_block! = ReweightBlock(ensemble_ancestral_proposal, 3)
    """Moves over babies"""
    baby_block! = ReweightBlock(baby_ancestral_proposal, 3)
end

function AuxState(::UniformProtocol)
    EmptyAuxState()
end

function apply_protocol!(chain::APChain, p::UniformProtocol)
    @unpack single_block!, ensemble_block!, baby_block! = p
    @unpack proc, state, auxillary = chain
    np = length(state.traces)
    for i = 1:np
        s = state.traces[i]
        # 1. Single
        k = s[:init_state => :k]
        winner = categorical(Fill(1.0 / k, k))
        single_block!(state, i, winner)

        # 2. Ensemble
        bernoulli(0.5) && ensemble_block!(state, i)

        # 3. Baby
        # bernoulli(0.5) && baby_block!(state, i)
    end

    return nothing
end

export AdaptiveProtocol

@with_kw struct AdaptiveProtocol <: AttentionProtocol
    """Moves over singles"""
    single_block! = ReweightBlock(single_ancestral_proposal, 10)
    """Moves over ensemble"""
    ensemble_block! = ReweightBlock(ensemble_ancestral_proposal, 3)
    """Moves over babies"""
    baby_block! = ReweightBlock(baby_ancestral_proposal, 3)
end

mutable struct AdaptiveAux <: AuxillaryState

end

function AuxState(::AdaptiveProtocol)
    AdaptiveAux()
end

function apply_protocol!(chain::APChain, p::AdaptiveProtocol)
    @unpack single_block!, ensemble_block!, baby_block! = p
    @unpack proc, state, auxillary = chain
    np = length(state.traces)
    for i = 1:np
        s = state.traces[i]
        # 1. Single
        k = s[:init_state => :k]
        winner = categorical(Fill(1.0 / k, k))
        single_block!(state, i, winner)

        # 2. Ensemble
        bernoulli(0.5) && ensemble_block!(state, i)

        # 3. Baby
        # bernoulli(0.5) && baby_block!(state, i)
    end

    return nothing
end
