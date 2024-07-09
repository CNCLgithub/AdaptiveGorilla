export AttentionProtocol,
    AuxState,
    apply_protocol!

abstract type AttentionProtocol end

"""
    AuxState(p::AttentionProtocol)

Initialize the associated auxillary state.
"""
function AuxState end

# TODO : document
function apply_protocol! end

include("proposals.jl")
include("particle_filter.jl")
include("attention_protocols.jl")
