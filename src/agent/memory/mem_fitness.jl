export MLLFitness, MhoFitness, CompFitness

################################################################################
# Marginal Log-likelihood Optimization
################################################################################

"""
Optimizes marginal log-likelihood across hyper particles
"""
struct MLLFitness <: MemoryFitness end

function init_fitness_state(::MLLFitness, chains, set_size)
    nothing
end

function memory_fitness_step!(::Nothing,
                              ::MLLFitness,
                              vis::MentalModule{V}
                              ) where {V<:HyperFilter}
    nothing
end

function memory_fitness_epoch!(::Nothing,
                               ::MLLFitness,
                               chain::APChain,
                               chain_idx::Int)
    log_ml_estimate(chain.state)
end

function update_fitness_reframe!(::Nothing, ::MLLFitness, t::Int,
                                 template::InertiaTrace, i::Int, parent::Int,
                                 cm::ChoiceMap)
    return nothing
end

function reset_state!(::Nothing)
    nothing
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
    "Overall exponential slope of complexity cost"
    complexity_mass::Float64 = 10.0
    "How sensitive cost is to a particular representation"
    complexity_factor::Float64 = 2.0
    "Rate of decay for delta time integral"
    log_decay_rate::Float64 = 0.0
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
        Vector{UInt64}(undef, chains),
        registry,
        Dict{UUID, Float64}()
    )
end

function memory_fitness_step!(mho::MhoScores,
                              gop::MhoFitness,
                              vis::MentalModule{V}
                              ) where {V<:HyperFilter}
    hf, vstate = mparse(vis)
    attp, attx = mparse(gop.att)

    map!(v -> v+(gop.log_decay_rate),
         values(mho.rep_deltas))

    increment = Dict{UUID, Float64}()
    counts = Dict{UUID, Int64}()

    for i = 1:hf.h
        chain = vstate.chains[i]
        schema_id = mho.schema_map[i]
        # Verify that the trace has not diverged from the schema
        # This could be due to birth/death moves.
        chain_map = retrieve_map(chain)
        if !is_valid_schema(mho.schema_registry, chain_map, schema_id)
            mho.schema_map[i] = schema_id =
                ammend_schema!(mho.schema_registry, chain_map, schema_id)
        end

        deltas = task_relevance(attx, attp.partition, chain_map, attp.nns)
        accumulate_deltas!(increment, counts, mho.schema_registry, schema_id,
                           deltas)
    end


    for (k, c) = counts
        increment[k] -= log(c)
    end
    # map!(v -> v - log(hf.h), values(increment))

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
    # print_granularity_schema(chain)
    # println(time_integral)
    # println("mho = $(round(mag; digits=2))(mag) - " *
    #     " $(round(irc;digits=2))(irc) = $(mho)")
    # println("--------------")
    return mho
end

function trace_mho(deltas::Vector{Float64}, temp::Float64, mass::Float64,
                   slope::Float64)
    mag = task_energy(deltas)
    importance = softmax(deltas, temp)
    c = irr_complexity(importance, mass, slope)
    (mag, c)
end

function task_energy(deltas::Vector{Float64})
    x = 0.0
    for d = deltas
        x += softplus(d)
    end
    return log(x)
end

function irr_complexity(imp::Vector{Float64}, mass::Float64, slope::Float64)
    n = length(imp)
    waste = 1E-4
    @inbounds for i = 1:n
        waste += exp(-slope * imp[i])
    end
    mass * log(waste)
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

function update_fitness_reframe!(mos::MhoScores, mof::MhoFitness, t::Int,
                                 template::InertiaTrace, i::Int, parent::Int,
                                 cm::ChoiceMap)
    schema_idx = mos.schema_map[parent]
    mos.new_schema_map[i] =
        transform_schema!(mos.schema_registry, schema_idx, t, template, cm)
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
    "Overall exponential slope of complexity cost"
    complexity_mass::Float64 = 0.5
end

function comp_complexity(trace::InertiaTrace,
                         mass::Float64)
    -Float64(representation_count(trace) / mass)
end

function init_fitness_state(::CompFitness, chains, set_size)
    nothing
end

function memory_fitness_step!(::Nothing,
                              ::CompFitness,
                              vis::MentalModule{V}
                              ) where {V<:HyperFilter}
    nothing
end

function memory_fitness_epoch!(::Nothing,
                               fit::CompFitness,
                               chain::APChain,
                               chain_idx::Int)
    chain_map = retrieve_map(chain)
    comp_complexity(chain_map, fit.complexity_mass)
end

function update_fitness_reframe!(::Nothing, ::CompFitness, t::Int,
                                 template::InertiaTrace, i::Int, parent::Int,
                                 cm::ChoiceMap)
    return nothing
end
