
"""
    perceive!(::MentalState{T}, ::T, ::Choicemap) where {T<:PerceptionProtocol}

Infer updates to mental representations
"""
function perceive! end

"""
    PerceptionModule(::T, ...)::MentalModule{T} where {T<:PerceptionProtocol}

Constructor that should be implemented by each `PerceptionProtocol`
"""
function PerceptionModule end

include("proposals.jl")
include("particle_filter.jl")
include("hyper_particle_protocol.jl")
