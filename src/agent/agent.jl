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


abstract type MentalProtocol end
abstract type MentalState{T<:MentalProtocol} end

mutable struct MentalModule{T<:MentalProtocol}
    protocol::T
    state::MentalState{T}
end

"""
Each protocol should implement this constructor
"""
function MentalModule end

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

abstract type PerceptionProtocol <: MentalProtocol end
abstract type PlanningProtocol <: MentalProtocol end
abstract type MemoryProtocol <: MentalProtocol end
abstract type AttentionProtocol <: MentalProtocol end


mutable struct Agent{
                     V<:PerceptionProtocol,
                     P<:PlanningProtocol,
                     M<:MemoryProtocol,
                     A<:AttentionProtocol
                    }
    perception::MentalModule{V}
    planning::MentalModule{P}
    memory::MentalModule{M}
    attention::MentalModule{A}
end


"""
    (TYPEDSIGNATURES)


"""
function agent_step!(agent::Agent, t::Int, obs::ChoiceMap)
    @unpack attention, perception, planning, memory = agent
    module_step!(perception, t, obs)
    module_step!(attention,  t, perception)
    module_step!(planning,   t, attention, perception)
    module_step!(memory,     t, perception)
    return nothing
end

function perceive!(agent::Agent, obs::ChoiceMap, t::Int)
    # perception runs every tick
    perceive!(agent.perception, obs)
    return nothing
end

function attend!(agent::Agent, t::Int)
    # attention runs every tick
    @unpack attention, perception = agent
    attend!(attention, perception)
    return nothing
end

function plan!(agent::Agent, t::Int)
    @unpack planning, attention, perception = agent
    plan!(planning, attention, perception, t)
    return nothing
end

function memory!(agent::Agent, t::Int)
    @unpack memory, attention, perception = agent
    # granularity assessment occurs every tick
    assess_memory!(memory, perception)
    # optimization occurs sparsely
    # (e.g., every 1.5s)
    optimize_memory!(memory, perception, t)
    return nothing
end

function perceive! end
function plan! end
function attend! end
function memory! end

# Mental module implementations
include("perception/perception.jl") # Hyper-particle filter
include("planning/planning.jl") # Event counting and planning-as-inference
include("attention/attention.jl") # Adaptive computation
include("memory/memory.jl") # Granularity optimizer

# agent-tailored visualizations
include("visuals.jl")
