# Benchmark: mRFS cost of iso elements vs. poisson elements over masks
#
# NOT an official test. Exploratory benchmark comparing the runtime / memory
# of evaluating `DetectionRFS` (mRFS over detection masks) with:
#   (1) 8 isomorphic elements (CIsoElement, one per object)
#   (2) 2-3 poisson elements (CPoissonElement, ensembles)
#
# Hypothesis: configuration (2) is slower than (1), despite having fewer
# elements, because poisson cardinalities blow up the partition enumeration.
#
# Run:  julia --project=. tests/bench_rfs_granularity.jl
# Note: BenchmarkTools must be in the project environment:
#   julia --project=. -e 'using Pkg; Pkg.add("BenchmarkTools")'

using Gen
using GenRFS
using Statistics
using LinearAlgebra
using BenchmarkTools
using AdaptiveGorilla
using StaticArrays: SVector

import AdaptiveGorilla as AG

using Profile
using StatProfilerHTML

################################################################################
# Command Line Interface
################################################################################

MODEL_VARIANTS = Dict(:mo => "Multi-Granular Optimization",
                      :ta => "Task-Agnostic Regranularization",
                      :ja => "Just Attention",
                      :fr => "Fixed Resource")


################################################################################
# Model Parameters
################################################################################

MODEL_PARAMS = "/project/scripts/study3/params"


################################################################################
# General Experiment Parameters
################################################################################

# which dataset to run
DATASET = "study3"
DPATH   = "/spaths/datasets/$(DATASET)/dataset.json"
SCENE   = 3
FRAMES  = 60

NTARGETS = 4
NDISTRACTORS = 6

################################################################################
# Methods
################################################################################

function run_model!(exp, model::String)
    model_path = "$(MODEL_PARAMS)/$(model).toml"
    agent = load_agent(model_path, exp.init_query)
    for t = 1:(FRAMES - 1)
        obs = AG.get_obs(exp, t)
        @profile AG.module_step!(agent.perception, t, obs)
        AG.module_step!(agent.attention,  t, agent.perception)
        AG.module_step!(agent.planning,   t, agent.attention, agent.perception)
        AG.module_step!(agent.memory,     t, agent.perception)
    end
    return nothing
end

################################################################################
# Main Entry
################################################################################

function main()
    result = NamedTuple[]
    nsteps = FRAMES-1
    # Load the world model
    wm = load_wm_from_toml("/project/scripts/study3/params/wm.toml";
                           object_rate = Float64(NTARGETS + NDISTRACTORS))
    # Load the experiment
    experiment = LoadCurve(wm, DPATH, SCENE, FRAMES, NTARGETS, NDISTRACTORS)

    Profile.clear()
    results = run_model!(experiment, "mo")

    open("profile.txt", "w") do io
        Profile.print(io; format = :tree, mincount = 10)
    end
    # display(Profile.print())
    statprofilehtml()
    return nothing
end;

main();

