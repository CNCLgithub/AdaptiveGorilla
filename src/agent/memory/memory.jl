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
    schema::Int64
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
    schema = memory_schema_set(perception)
    HyperResampling(prot.h, schema, fitness, kernel, tau, schema_log_decay_rate)
end

mutable struct MemoryAssessments <: MentalState{HyperResampling}
    chain_objectives::Vector{Float64}
    schema_registry::SchemaRegistry
    schema_map::Vector{UInt64}
    new_schema_map::Vector{UInt64}
    rep_objectives::Dict{UUID, Float64}
    steps::Int
end

function MemoryAssessments(chains::Int, schema_set::Int64)
    registry = SchemaRegistry(schema_set)
    ischema = init_schema(registry)
    MemoryAssessments(fill(-Inf, chains),
                      registry,
                      fill(ischema, chains),
                      Vector{UInt64}(undef, chains),
                      Dict{UUID, Float64}(),
                      0)
end

# TODO: document
function MemoryModule(p::HyperResampling)
    MentalModule(p, MemoryAssessments(p.chains, p.schema))
end

function module_step!(mem::MentalModule{M},
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
        prev = mstate.chain_objectives[i]
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
         values(mstate.rep_objectives))

    # Init integral step
    rep_counts = Dict{UUID, Int}()
    rep_acc = Dict{UUID, Float64}()
    
    for i = 1:hf.h
        # obtain schema score for t
        chain = vstate.chains[i]
        schema_score = mstate.chain_objectives[i] -= log(mstate.steps)


        schema_id = mstate.schema_map[i]
        # Verify that the trace has not diverged from the schema
        # This could be due to birth/death moves.
        chain_map = retrieve_map(chain)
        if !is_valid_schema(mstate.schema_registry, chain_map, schema_id)
            mstate.schema_map[i] = schema_id =
                ammend_schema!(mstate.schema_registry, chain_map, schema_id)
        end

        # integral from 0 to t-1
        time_integral = aggregate_scores(mstate.rep_objectives,
                                         mstate.schema_registry,
                                         schema_id,
                                         schema_score)

        # println(@sprintf "Schema %d score: %4f.1, ⨍ %4f.1" i schema_score time_integral)

        # store time integrated objective for the chain
        mstate.chain_objectives[i] = logsumexp(schema_score, time_integral)

        # accumulate current score for next time integral
        distribute_scores!(rep_acc, rep_counts, mstate.schema_registry,
                           schema_id, schema_score)
    end

    # Normalize schema time integrals
    for (k, c) = rep_counts
        rep_acc[k] -= log(hf.h)
    end

    # Merge and update schema record
    mergewith!(logsumexp, mstate.rep_objectives, rep_acc)

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

    # println()
    # println("\t################################################# ")
    # println("\t#              | CHAIN WEIGHTS |                # ")
    # println("\t#              |  FRAME: $(t)  |                # ")
    # println("\t################################################# ")
    # plot_rep_weights(memstate.schema_registry, memstate.rep_objectives)
    # for i = 1:visp.h
    #     print_granularity_schema(visstate.chains[i])
    #     println("\t OBJ: $(memstate.chain_objectives[i]) \n\t W: $(ws[i])")
    # end
    # @show next_gen
    # println("\t === Top Schema ===")
    # describe_schema(memstate.schema_registry, memstate.schema_map[argmax(ws)])

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
        cm = restructure_kernel(memp.kernel, template)
        # println("Reframing hyper particle $(i) to:")
        # display(cm)
        # Step 3
        # REVIEW: copy over parent schema score?
        schema_idx = memstate.schema_map[parent]
        memstate.new_schema_map[i] =
            transform_schema!(memstate.schema_registry, schema_idx, t,
                              template, cm)
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
    fill!(memstate.chain_objectives, -Inf)
    memstate.steps = 0
    # swap references
    temp = memstate.schema_map
    memstate.schema_map = memstate.new_schema_map
    memstate.new_schema_map = temp
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

