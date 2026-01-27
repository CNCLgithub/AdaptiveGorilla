export MemoryModule,
    HyperResampling,
    MemoryAssessments,
    MemoryFitness,
    memory_fitness,
    RestructuringKernel,
    restructure_kernel

include("schema.jl")

################################################################################
# Memory Protocols
################################################################################

"Defines the particular objective optimizing hyper chains"
abstract type MemoryFitness end

"Defines how memory format alters"
abstract type RestructuringKernel end


"""
    $(TYPEDEF)

Resamples [`HyperFilter`](@ref) chains.

### Fields

$(TYPEDFIELDS)
"""
struct HyperResampling <: MemoryProtocol
    "Number of hyper chains in perception"
    chains::Int64
    "Initial Schema Set"
    schema::Int64 # HACK: this is passed to init fitness_state
    "Optimization fitness criteria"
    fitness::MemoryFitness
    "Restructuring Kernel"
    kernel::RestructuringKernel
    "Temperature for chain resampling"
    tau::Float64

end

function HyperResampling(; perception::MentalModule{<:HyperFilter},
                         fitness, kernel, tau = 1.0)
    prot, _ = mparse(perception)
    schema = memory_schema_set(perception)
    HyperResampling(prot.h, schema, fitness, kernel, tau)
end

mutable struct MemoryAssessments{T} <: MentalState{HyperResampling}
    fitness_state::T
    chain_objectives::Vector{Float64}
end

function MemoryAssessments(fitness::MemoryFitness, chains::Int, schema_set::Int64)
    state = init_fitness_state(fitness, chains, schema_set)
    MemoryAssessments(state, fill(-Inf, chains))
end

# TODO: document
function MemoryModule(p::HyperResampling)
    MentalModule(p, MemoryAssessments(p.fitness, p.chains, p.schema))
end



function module_step!(mem::MentalModule{M},
                      t::Int,
                      vis::MentalModule{V}
                      ) where {M<:HyperResampling,
                               V<:HyperFilter}
    memory_fitness_step!(mem, t, vis)
    optimize_memory!(mem, t, vis)
    return nothing
end

function memory_fitness_step!(mem::MentalModule{M},
                              t::Int,
                              vis::MentalModule{V}
                              ) where {M<:MemoryProtocol,
                                       V<:PerceptionProtocol}
    mp, ms = mparse(mem)
    memory_fitness_step!(ms.fitness_state, mp.fitness, vis)
end


function assess_memory_epoch!(mem::MentalModule{M},
                              t::Int,
                              vis::MentalModule{V}
                              ) where {M<:HyperResampling,
                                       V<:HyperFilter}
    mp, ms = mparse(mem)
    hf, vstate = mparse(vis)
    for i = 1:hf.h
        chain = vstate.chains[i]
        ms.chain_objectives[i] =
            memory_fitness_epoch!(ms.fitness_state, mp.fitness, chain, i)
    end
    return nothing
end

function optimize_memory!(mem::MentalModule{M},
                          t::Int,
                          vis::MentalModule{V}
    ) where {M<:HyperResampling,
             V<:HyperFilter}

    visp, visstate = mparse(vis)
    memp, memstate = mparse(mem)
     
    # Not time yet for reframing
    (t > 0 && t % visp.dt != 0) && return nothing

    assess_memory_epoch!(mem, t, vis)

    # Estimated sample size
    lognormed_objectives = memstate.chain_objectives .-
        logsumexp(memstate.chain_objectives)
    ess = Gen.effective_sample_size(lognormed_objectives)

    # Maybe resample
    ws = softmax(memstate.chain_objectives, memp.tau)
    next_gen = Vector{Int}(undef, visp.h)
    residual_resample!(next_gen, ws)
    # if ess <= 0.5 * visp.h
    #     Distributions.rand!(Distributions.Categorical(ws), next_gen)
    # else
    #     next_gen[:] = 1:visp.h
    # end

    # println("\n\t################################################# ")
    # println("\t#              | CHAIN WEIGHTS |                # ")
    # println("\t#              |  FRAME: $(t)  |                # ")
    # println("\t################################################# ")
    # plot_fitness(memstate.fitness_state)
    # for i = 1:visp.h
    #     print_granularity_schema(visstate.chains[i])
    #     println("\t OBJ: $(memstate.chain_objectives[i]) \n\t W: $(ws[i])")
    # end
    # @show next_gen
    # println("\t === Top Schema ===")
    # describe_chain_fitness(memstate.fitness_state, argmax(ws))

    # For each hyper particle:
    # 1. extract the MAP as a seed trace for the next generation
    # 2. sample a change in memory structure (optional)
    # 3. Re-initialized hyper particle chain from seed.
    for i = 1:visp.h
        # Step 1
        parent = next_gen[i]
        chain = visstate.chains[parent]
        template = retrieve_map(chain) # InertiaTrace
        # Step 2 (optional)
        cm = restructure_kernel(memp.kernel, memp.fitness,
                                memstate.fitness_state, template, parent)

        # println("Reframing hyper particle $(i) to:")
        # display(cm)
         
        # Step 3
        update_fitness_reframe!(memstate.fitness_state, memp.fitness, t,
                                template, i, parent, cm)
        visstate.new_chains[i] = reinit_chain(chain, template, cm)
    end

    # Set perception fields
    reset_state!(visstate, visp)
    # Reset optimizer state
    reset_state!(memstate, memp)
    
    return nothing
end

function reset_state!(memstate::MemoryAssessments, memp::HyperResampling)
    # clear instantaneous objectives
    # fill!(memstate.chain_objectives, -Inf)

    reset_state!(memstate.fitness_state)
    return nothing
end

function residual_resample!(next::Vector{Int64},
                            ws::Vector{Float64},
                            thresh_factor::Float64 = 0.5)
    n = length(next)
    thresh = thresh_factor / n
    to_resample = ws .< thresh
    ws[to_resample] .= 0.0
    rmul!(ws, 1.0 / sum(ws))
    for i = 1:n
        next[i] = to_resample[i] ? categorical(ws) : i
    end
    nothing
end

################################################################################
# Memory Optimizers
################################################################################


"""
    memory_fitness(optim, att, vision)

Returns the fitness of a hyper particle.
"""
function memory_fitness end


""""
    init_fitness_state(fit, chains, schema_set_size)

Initializes the auxillary state (if any) used in approximating fitness.
"""
function init_fitness_state end

"""
    memory_fitness_step!(mem, t, vis)

Increments fitness calculations every time step.
"""
function memory_fitness_step! end


"""
    memory_fitness_epoch!(fit_state, fit_proc, chain, chain_idx)

Determines the objectives for each chain. Will be used for resampling.
"""
function memory_fitness_epoch! end

include("mem_fitness.jl")


################################################################################
# Restructuring Kernels
################################################################################

"""

    restructure_kernel(kernel, trace)

Samples a choicemap that alters the granularity schema of `trace`.
"""
function restructure_kernel end

include("restructuring_kernel.jl")

