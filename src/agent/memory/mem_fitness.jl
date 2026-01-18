export MLLFitness, MhoFitness, CompFitness

################################################################################
# Marginal Log-likelihood Optimization
################################################################################

"""
Optimizes marginal log-likelihood across hyper particles
"""
struct MLLFitness <: MemoryFitness end

function memory_fitness(optim::MLLFitness,
                        chain::APChain)
    log_ml_estimate(chain.state)
end

################################################################################
# Granularity Optimization
################################################################################

"""
Defines the granularity mapping for a current trace format
"""
@with_kw struct MhoFitness <: MemoryFitness
    "Attention required for task-relevance"
    att::MentalModule{<:AdaptiveComputation}
    "Log-scaling factor for MLL"
    beta::Float64 = 5.0
    "Overall exponential slope of complexity cost"
    complexity_mass::Float64 = 10.0
    "How sensitive cost is to a particular representation"
    complexity_factor::Float64 = 2.0
end

mutable struct MhoScores
    schema_map::Vector{UInt64}
    new_schema_map::Vector{UInt64}
    schema_registry::SchemaRegistry
    rep_deltas::Dict{UUID, Float64}
end


function init_fitness_state(f::MhoFitness, chains::Int, schema_set_size::Int)
    registry = SchemaRegistry(schema_set_size)
    ischema = init_schema(registry)
    MhoScores(
        fill(ischema, chains),
        fill(-Inf, chains),
        Vector{UInt64}(undef, chains),
        registry,
        Dict{UUID, Float64}()
    )
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

function assess_memory_step!(mem::MentalModule{M},
                             t::Int,
                             vis::MentalModule{V}
                             ) where {M<:HyperResampling,
                                      V<:HyperFilter}
    mp, mstate = mparse(mem)
    memory_fitness_step!(mp.fitness_state, mp.fitness, vis)
end

function memory_fitness_step!(mho::MhoScores,
                              gop::MhoFitness,
                              vis::MentalModule{V}
                              ) where {V<:HyperFilter}
    hf, vstate = mparse(vis)
    attp, attx = mparse(gop.att)

    map!(v -> v+(mho.log_decay_rate),
         values(mho.rep_deltas))

    increment = Dict{UUID, Float64}()

    for i = 1:hf.h
        chain = vstate.chains[i]
        # Verify that the trace has not diverged from the schema
        # This could be due to birth/death moves.
        chain_map = retrieve_map(chain)
        if !is_valid_schema(mstate.schema_registry, chain_map, schema_id)
            mho.schema_map[i] = schema_id =
                ammend_schema!(mho.schema_registry, chain_map, schema_id)
        end

        deltas = task_relevance(attx, attp.partition, chain_map, attp.nns)

        accumulate_deltas!(increment, mho.schema_registry, schema_id)
    end


    map!(v -> v - log(hf.h), values(increment))

    # Merge and update
    mergewith!(logsumexp, mho.rep_deltas, increment)

    return nothing
end

function memory_fitness_epoch!(fit_state::MhoScores,
                               fit_proc::MhoFitness,
                               chain::APChain,
                               chain_idx::Int)

    attp, attx = mparse(fit_proc.att)

    # integral from 0 to t
    schema_id = fit_state.schema_map[chain_idx]
    time_integral = reconstitute_deltas(fit_state.rep_deltas,
                                        fit_state.schema_registry,
                                        schema_id)
    (mag, irc) = trace_mho(time_integral, attp.itemp,
                           fit_proc.complexity_mass,
                           fit_proc.complexity_factor)

    mho = mag - irc

    lml = log_ml_estimate(chain.state) / fit_proc.beta

    print_granularity_schema(chain)
    println(time_integral)
    println("mho = $(round(mag; digits=2))(mag) - " *
        " $(round(irc;digits=2))(irc) = $(mho)")
    @show lml
    @show mho + lml
    println("--------------")
    
    mho + lml
end

function trace_mho(deltas::Vector{Float64},
                   temp::Float64,
                   mass::Float64,
                   slope::Float64)

    mag = logsumexp(temp)
    importance = softmax(deltas, temp)
    c = irr_complexity(importance, mass, slope)
    (mag, c)
end

function irr_complexity(imp::Vector{Float64}, mass::Float64, slope::Float64)
    n = length(imp)
    waste = 0.0
    @inbounds for i = 1:n
        # w = mass * (1 - imp[i])^(factor)
        w = mass * exp(-slope * imp[i])
        # println("i $(imp[i]) -> w $(w)")
        waste += w
    end
    # pad, will be denominator
    waste + 1E-4
end

function print_granularity_schema(chain::APChain)
    tr = retrieve_map(chain)
    print_granularity_schema(tr)
end

function print_granularity_schema(tr::InertiaTrace)
    print_granularity_schema(get_last_state(tr))
end

function plot_fitness(m::MhoScores)
    plot_rep_weights(m.schema_registry, m.rep_deltas)
end

function describe_chain_fitness(m::MhoScores, chain_idx::Int)
    describe_schema(m.schema_registry, m.schema_map[chain_idx])
end

function update_fitness_reframe!(mos::MhoScores, mof::MhoFitness,
                                 t::Int, i::Int, parent::Int,
                                 cm::ChoiceMap)
    schema_idx = mos.schema_map[parent]
    mos.new_schema_map[i] =
        transform_schema!(ms.schema_registry, schema_idx, t, template, cm)
    return nothing
end

function reset_state!(mos::MhoScores)
    # swap references
    temp = mos.schema_map
    mos.schema_map = mos.new_schema_map
    mos.new_schema_map = temp
    return nothing
end

################################################################################
# Computational Complexity
################################################################################

@with_kw struct CompFitness <: MemoryFitness
    "Log-scaling factor for MLL"
    beta::Float64 = 5.0
    "Overall exponential slope of complexity cost"
    complexity_mass::Float64 = 0.5
end

function memory_fitness(gop::CompFitness,
                        chain::APChain)
    @unpack state = chain
    lml = log_ml_estimate(state) / gop.beta
    # average across particles
    nparticles = length(state.traces)
    compv = Vector{Float64}(undef, nparticles)
    ircv = Vector{Float64}(undef, nparticles)
    @inbounds for i = 1:nparticles
        compv[i] = comp_complexity(state.traces[i], gop.complexity_mass)
    end

    mag = logsumexp(compv) - log(nparticles)
    mag + lml
end

function comp_complexity(trace::InertiaTrace,
                         mass::Float64)
    -Float64(representation_count(trace) / mass)
end
