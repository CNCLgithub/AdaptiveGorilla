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
using Accessors
using Parameters
using FillArrays
using Statistics
using StaticArrays
using LinearAlgebra
using DocStringExtensions
using FunctionalCollections

include("utils/utils.jl")
include("dgp/dgp.jl")
include("gm/gm.jl")
include("planner/planner.jl")
include("inference/inference.jl")




end # module AdaptiveGorilla
