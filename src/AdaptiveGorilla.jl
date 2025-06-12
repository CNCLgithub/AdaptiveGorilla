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
using Gen_Compose
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


include("agent/agent.jl")

# TODO: finish moving
include("experiments/experiments.jl")

end # module AdaptiveGorilla
