"""
Implementation of \"It's a feature not a bug: Multi-granular world models explain inattentional blindness\"

# Citation

```bib
@article{belledonne_feature,
  author   = \"Belledonne, Mario and Yildirim, Ilker\",
  title    = \"It's a feature not a bug: Multi-granular world models explain inattentional blindness\",
  journal  = \"TBD\",
  year     = TBD,
  volume   = \"TBD\",
  number   = \"TBD\",
  pages    = \"TBD\",
}
```

---

# Exports

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
include("gm/gm.jl")
include("agent/agent.jl")
include("experiments/experiments.jl")

end # module AdaptiveGorilla
