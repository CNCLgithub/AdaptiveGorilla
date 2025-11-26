export load_agent

function load_perception!(parts, toml, query)
    stub = toml["Perception"]
    protocol = getfield(AdaptiveGorilla, Symbol(stub["protocol"]))
    kwargs = load_inner(parts, stub["params"])
    parts[:Perception] = PerceptionModule(protocol(; kwargs...), query)
    return nothing
end

function load_planning!(parts, toml)
    stub = toml["Planning"]
    protocol = getfield(AdaptiveGorilla, Symbol(stub["protocol"]))
    kwargs = load_inner(parts, stub["params"])
    parts[:Planning] = PlanningModule(protocol(; kwargs...))
    return nothing
end


function load_attention!(parts, toml)
    stub = toml["Attention"]
    protocol = getfield(AdaptiveGorilla, Symbol(stub["protocol"]))
    kwargs = load_inner(parts, stub["params"])
    parts[:Attention] = AttentionModule(protocol(; kwargs...))
    return nothing
end


function load_memory!(parts, toml)
    stub = toml["Memory"]
    protocol = getfield(AdaptiveGorilla, Symbol(stub["protocol"]))
    kwargs = load_inner(parts, stub["params"])
    parts[:Memory] = MemoryModule(protocol(; kwargs...))
    return nothing
end

function load_from_parts(parts)
    Agent(
        parts[:Perception],
        parts[:Planning],
        parts[:Memory],
        parts[:Attention],
    )
end

"""
    $(TYPEDSIGNATURES)

Instantiates an agent from a TOML file
"""
function load_agent(path::String, query)
    toml = TOML.parsefile(path)
    get(toml, "format", nothing) == "Agent" || error("Not valid format")
    parts = Dict()
    load_perception!(parts, toml, query)
    load_planning!(parts, toml)
    load_attention!(parts, toml)
    load_memory!(parts, toml)
    load_from_parts(parts)
end
