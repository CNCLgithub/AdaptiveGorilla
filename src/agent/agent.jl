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

#TODO: Doc me!
function mparse(m::MentalModule)
    (m.protocol, m.state)
end

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
    assess_granularity!(memory, attention, perception)
    # regranularization occurs sparsely
    regranularize!(memory, attention, perception, t)
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
