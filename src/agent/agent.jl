export perceive!, Agent, MentalProtocol,
    MentalState, MentalModule, mparse,
    PerceptionProtocol,
    PlanningProtocol,
    MemoryProtocol,
    AttentionProtocol,
    plan!,
    attend!,
    regranulize!,
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

world_model(a::Agent) = a.world_model

# TODO: dispatch
function perceive!(agent::Agent, obs::ChoiceMap)
    @unpack perception, attention = agent
    perceive!(perception, attention, obs)
    return nothing
end

function plan!(agent::Agent)
    @unpack (memory, perception, planning,
             attention) = agent
    plan_in = transfer(memory, perception)
    plan!(planning, attention, plan_in)
    return nothing
end

function perceive! end
function plan! end
function attend! end
function regranulize! end

include("perception/perception.jl")
include("planning.jl")
include("attention/attention.jl")
include("memory/memory.jl")

function start_exp(exp::Gorillas)
end

function step_agent!(agent::Agent, exp::Gorillas)

    obs = get_obs(exp, step)
    # S, attention, dS
    perceive!(agent, obs)
    # Pi, dPi
    plan!(agent)
    # New granularity
    regranulize!(agent)

end
