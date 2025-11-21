export PlanningProtocol,
    PlanningModule

"""

    PlanningModule(::T, ...)::MentalModule{T} where {T<:PlanningProtocol}

Constructor that should be implemented by each `PlanningProtocol`

---

# Implementations:

(METHODLIST)
"""
function PlanningModule end

include("collision.jl")
# include("safari.jl")
