export PlanningProtocol,
    PlanningModule,
    plan!

"""
A MentalProtocol for planning.

"""
abstract type PlanningProtocol <: MentalProtocol end

"""

    PlanningModule(::T, ...)::MentalModule{T} where {T<:PlanningProtocol}

Constructor that should be implemented by each `PlanningProtocol`

---

# Implementations:

(METHODLIST)
"""
function PlanningModule end

"""

    plan!(planner::MentalModule{T},
          attention::MentalModule{A},
          perception::MentalModule{V},
          t::Int
         ) where {T<:PlanningProtocol,
                  A<:AttentionProtocol,
                  V<:PerceptionProtocol}

Increment planning module.

Receives the perception module as input and (optionally)
interfaces with attention.

---

# Implementations

$(METHODLIST)
"""
function plan! end

include("collision.jl")
# include("safari.jl")
