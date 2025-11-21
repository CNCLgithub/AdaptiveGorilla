export Agent, MentalProtocol,
    MentalState, MentalModule, mparse,
    PerceptionProtocol,
    PlanningProtocol,
    MemoryProtocol,
    AttentionProtocol,
    perceive!,
    plan!,
    attend!,
    memory!,
    step_agent!


"The algorithmic implementation of a mental process"
abstract type MentalProtocol end

"The state of a mental process"
abstract type MentalState{T<:MentalProtocol} end

"""
$(TYPEDEF)

An algorithmic "organ" - both its function (protocol) and form (state).

---

Each protocol should call this constructor. See [`PerceptionModule`](@ref)

"""
mutable struct MentalModule{T<:MentalProtocol}
    protocol::T
    state::MentalState{T}
end


"""
    $(TYPEDSIGNATURES)

Returns the protocol and state of a mental module.
"""
function mparse(m::MentalModule)
    (m.protocol, m.state)
end


"""
    $(FUNCTIONNAME)(module, t, ...)

Run the mental module forward for one tick.

Here, `t::Int` denotes the global clock step.
Not all modules will perform operations every tick.
Should return `Nothing`.

---

$(METHODLIST)
"""
function module_step! end

"How to generate world models from observations"
abstract type PerceptionProtocol <: MentalProtocol end

"How to generate decisions from percepts"
abstract type PlanningProtocol <: MentalProtocol end

"A Frame is worth 1000 words"
abstract type MemoryProtocol <: MentalProtocol end

"Attending across time and space"
abstract type AttentionProtocol <: MentalProtocol end


"""
    $(TYPEDEF)

A simulated agent.

---

$(TYPEDFIELDS)

"""
mutable struct Agent{
                     V<:PerceptionProtocol,
                     P<:PlanningProtocol,
                     M<:MemoryProtocol,
                     A<:AttentionProtocol
                    }
    "Observations -> Worlds"
    perception::MentalModule{V}
    "Worlds -> Goals"
    planning::MentalModule{P}
    "How to frame the world"
    memory::MentalModule{M}
    "What to attend to in the world"
    attention::MentalModule{A}
end


"""
    (TYPEDSIGNATURES)

Simulate one tick in the agent's mind.

Not all modules will necessarily operate at each tick.
"""
function agent_step!(agent::Agent, t::Int, obs::ChoiceMap)
    @unpack attention, perception, planning, memory = agent
    module_step!(perception, t, obs)
    module_step!(attention,  t, perception)
    module_step!(planning,   t, attention, perception)
    module_step!(memory,     t, perception)
    return nothing
end

# Mental module implementations
include("perception/perception.jl") # Hyper-particle filter
include("planning/planning.jl") # Event counting and planning-as-inference
include("attention/attention.jl") # Adaptive computation
include("memory/memory.jl") # Granularity optimizer

# agent-tailored visualizations
include("visuals.jl")
include("io.jl")
