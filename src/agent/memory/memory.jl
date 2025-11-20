export (MemoryModule,
        assess_memory!,
        optimize_memory!,
        HyperResampling,
        MemoryAssessments)

################################################################################
# Memory Protocols
################################################################################

@with_kw struct HyperResampling <: MemoryProtocol
    "Temperature for chain resampling"
    tau::Float64 = 1.0
    "Optimization fitness criteria"
    fitness::MemoryFitness = MLLFitness()
    "Restructuring Kernel"
    kernel::RestructuringKernel = StaticRKernel()
end

mutable struct MemoryAssessments <: MentalState{GranularityOptim}
    objectives::Vector{Float64}
    steps::Int
end

function MemoryAssessments(size::Int)
    MemoryAssessments(zeros(size), 0)
end

# TODO: document
function MemoryModule(p::GranularityOptim, size::Int)
    MentalModule(p, MemoryAssessments(size))
end

function assess_memory!(
    mem::MentalModule{M},
    vis::MentalModule{V}
    ) where {M<:GranularityOptim,
             V<:HyperFilter}
    mp, mstate = mparse(mem)
    hf, vstate = mparse(vis)
    for i = 1:hf.h
        increment = memory_fitness(mp.optim, vstate.chains[i])
        mstate.objectives[i] = logsumexp(mstate.objectives[i], increment)
    end
    mstate.steps += 1
    return nothing
end

function optimize_memory!(mem::MentalModule{M},
                          vis::MentalModule{V},
                          t::Int
    ) where {M<:GranularityOptim,
             V<:HyperFilter}

    memp, memstate = mparse(mem)
    visp, visstate = mparse(vis)
    # not time yet
    (t > 0 && t % visp.dt != 0) && return nothing

    # Repopulate and potentially alter memory schemas
    memstate.objectives .-= log(memstate.steps)
    ws = softmax(memstate.objectives, memp.tau)
    next_gen = Vector{Int}(undef, visp.h)
    Distributions.rand!(Distributions.Categorical(ws), next_gen)

    # For each hyper particle:
    # 1. extract the MAP as a seed trace for the next generation
    # 2. sample a change in memory structure (optional)
    # 3. Re-initialized hyper particle chain from seed.
    for i = 1:visp.h
        parent = visstate.chains[next_gen[i]] # PFChain
        template = retrieve_map(parent) # InertiaTrace
        cm = restructure_kernel(memp.kernel, template)
        visstate.new_chains[i] = reinit_chain(parent, template, cm)
    end

    # Set perception fields
    visstate.age = 1
    temp_chains = visstate.chains
    visstate.chains = visstate.new_chains
    visstate.new_chains = temp_chains

    # Reset optimizer state
    fill!(memstate.objectives, -Inf)
    memstate.steps = 0

    return nothing
end


################################################################################
# Restructuring Kernels
################################################################################

"Defines how memory format alters"
abstract type RestructuringKernel end

"""

    restructure_kernel(kernel, trace)

Samples a choicemap that alters the granularity schema of `trace`.
"""
function restructure_kernel end

include("restructuring_kernel.jl"

################################################################################
# Memory Optimizers
################################################################################

"Defines the particular objective optimizing hyper chains"
abstract type MemoryFitness end


"""
    memory_fitness(optim, att, vision)

Returns the fitness of a hyper particle.
"""
function memory_fitness end

"""
    restructure_kernel(optim, att, template::Trace)

Returns a choicemap that alters the memory schema of a trace.
"""
function restructure_kernel end

include("mem_fitness.jl")

# function print_granularity_schema(chain::APChain)
#     tr = retrieve_map(chain)
#     print_granularity_schema(tr)
# end
# function print_granularity_schema(tr::InertiaTrace)
#     state = get_last_state(tr)
#     ns = length(state.singles)
#     ne = length(state.ensembles)
#     c = object_count(tr)
#     println("MAP Granularity: $(ns) singles; $(ne) ensembles; $(c) total")
#     ndark = count(x -> material(x) == Dark, state.singles)
#     println("\tSingles: $(ndark) Dark | $(ns-ndark) Light")
#     println("\tEnsembles: $(map(e -> (rate(e), e.matws[1]), state.ensembles))")
#     return nothing
# end
