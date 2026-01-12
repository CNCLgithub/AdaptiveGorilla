export MemoryModule,
    HyperResampling,
    MemoryAssessments,
    MemoryFitness,
    memory_fitness,
    RestructuringKernel,
    restructure_kernel

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
    "Optimization fitness criteria"
    fitness::MemoryFitness
    "Restructuring Kernel"
    kernel::RestructuringKernel
    "Temperature for chain resampling"
    tau::Float64
    schema_log_decay_rate::Float64

end

function HyperResampling(; perception::MentalModule{<:HyperFilter},
                         fitness, kernel, tau = 1.0,
                         schema_log_decay_rate = -Inf)
    prot, _ = mparse(perception)
    HyperResampling(prot.h, fitness, kernel, tau, schema_log_decay_rate)
end

mutable struct MemoryAssessments <: MentalState{HyperResampling}
    chain_objectives::Vector{Float64}
    schema_map::Vector{Int}
    schema_objectives::Dict{Int, Float64}
    steps::Int
end

function MemoryAssessments(size::Int)
    # REVIEW: init of `schema_map` may be dangerous
    MemoryAssessments(zeros(size), zeros(UInt, size), Dict{UInt, Float64}(), 0)
end

# TODO: document
function MemoryModule(p::HyperResampling)
    MentalModule(p, MemoryAssessments(p.chains))
end

function module_step!(
    mem::MentalModule{M},
    t::Int,
    vis::MentalModule{V}
    ) where {M<:HyperResampling,
             V<:HyperFilter}
    assess_memory_step!(mem, t, vis)
    optimize_memory!(mem, t, vis)
    return nothing
end

function assess_memory_step!(mem::MentalModule{M},
                        t::Int,
                        vis::MentalModule{V}
                        ) where {M<:HyperResampling,
                                 V<:HyperFilter}
    mp, mstate = mparse(mem)
    hf, vstate = mparse(vis)
    for i = 1:hf.h
        chain = vstate.chains[i]
        increment = memory_fitness(mp.fitness, chain)
        mstate.chain_objectives[i] = logsumexp(prev, increment)
    end
    mstate.steps += 1
    return nothing
end

function assess_memory_epoch!(mem::MentalModule{M},
                              t::Int,
                              vis::MentalModule{V}
                              ) where {M<:HyperResampling,
                                       V<:HyperFilter}
    mp, mstate = mparse(mem)
    hf, vstate = mparse(vis)

    # Decay schema objectives
    map!(v -> v+(mp.schema_log_decay_rate),
         values(mstate.schema_objectives))


    schema_chain_counts = Dict{Int, Int}()
    schema_acc = Dict{Int, Float64}()
    
    for i = 1:hf.h
        chain = vstate.chains[i]
        obj = mstate.chain_objectives[i] -= m.steps

        schema = mstate.schema_map[i]
        # REVIEW: What if there is a birth at `t`?
        chain_map = retrieve_map(chain)
        if !is_valid_schema(chain_map, schema)
            mstate.schema_map[i] = schema =
                ammend_schema(chain_map, schema)
        end

        # store time integrated objective for the chain
        mstate.objectives[i] =
            logsumexp(obj, get(mstate.schema_objectives, schema, -Inf))

        # accumulate for time integral
        schema_acc[schema] = logsumexp(get(schema_acc, schema, -Inf), obj)
        schema_chain_counts[schema] = get(schema_chain_count, schema, 0) + 1
    end

    # Normalize schema time integrals
    for (k, c) = schema_chain_counts
        schema_acc[k] -= log(c)
    end

    # Merge and update schema record
    mergewith!(logsumexp, mstate.schema_objectives, new_schema_objectives)

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
    lognormed_objectives = mstate.objectives .- logsumexp(mstate.objectives)
    ess = Gen.effective_sample_size(lognormed_objectives)

    # Maybe resample
    ws = softmax(memstate.objectives, memp.tau)
    next_gen = Vector{Int}(undef, visp.h)
    if ess < 0.5 * visp.h
        Distributions.rand!(Distributions.Categorical(ws), next_gen)
    else
        next_gen[:] = 1:visp.h
    end

    println()
    println("################################################# ")
    println("#________________CHAIN WEIGHTS__________________# ")
    println("#________________FRAME: $(t)    ________________# ")
    println("################################################# ")
    attp, attx = mparse(memp.fitness.att)
    for i = 1:visp.h
        print_granularity_schema(visstate.chains[i])
        println("OBJ: $(memstate.objectives[i]) \n W: $(ws[i])")
        tr = task_relevance(attx,
                            attp.partition,
                            retrieve_map(visstate.chains[i]),
                            attp.nns)
        @show tr 
    mag = logsumexp(tr)
    end
    @show next_gen

    # For each hyper particle:
    # 1. extract the MAP as a seed trace for the next generation
    # 2. sample a change in memory structure (optional)
    # 3. Re-initialized hyper particle chain from seed.
    for i = 1:visp.h
        # Step 1
        parent = next_gen[i]
        chain = visstate.chains[i]
        template = retrieve_map(chain) # InertiaTrace
        # Step 2 (optional)
        cm = restructure_kernel(memp.kernel, template)
        # Step 3
        # REVIEW: copy over parent schema score? 
        mstate.schema_map[i] = transform_schema(schema, cm) # TODO: implement `transform_schema`
        visstate.new_chains[i] = reinit_chain(parent, template, cm)
    end

    # Set perception fields
    reset_state!(visstate, visp)
    # Reset optimizer state
    reset_state!(memstate, memp)
    
    return nothing
end

# TODO: integrate changes
function reset_state!(memstate::MemoryAssessments, memp::HyperResampling)
    fill!(memstate.objectives, -Inf)
    memstate.steps = 0
end

################################################################################
# Memory Optimizers
################################################################################


"""
    memory_fitness(optim, att, vision)

Returns the fitness of a hyper particle.
"""
function memory_fitness end


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

