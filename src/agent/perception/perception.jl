"""
    PerceptionModule(::T, ...)::MentalModule{T} where {T<:PerceptionProtocol}

Constructor that should be implemented by each `PerceptionProtocol`

## Implementations:

$(METHODLIST)

"""
function PerceptionModule end

include("proposals.jl")
include("particle_filter.jl")
include("hyper_particle_protocol.jl")
