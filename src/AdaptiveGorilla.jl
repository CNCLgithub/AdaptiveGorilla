"""
A model of inattentional blindness.

---

EXPORTS:

$(EXPORTS)


"""
module AdaptiveGorilla

using Gen
using JSON3
using GenRFS
using MOTCore
using Distances
using Accessors
using FillArrays
using Parameters
using Statistics
using StaticArrays
using Distributions
using LinearAlgebra
using DataStructures
using NearestNeighbors
using DocStringExtensions
using FunctionalCollections

include("utils/utils.jl")
include("dgp/dgp.jl")
include("gm/gm.jl")

include("experiment.jl")

include("agent/agent.jl")

include("planner/planner.jl")
include("inference/inference.jl")




end # module AdaptiveGorilla
