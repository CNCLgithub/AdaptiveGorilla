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


function memory_fitness(gop::MhoFitness,
                        chain::APChain)
    attp, attx = mparse(gop.att)
    @unpack state = chain
    lml = log_ml_estimate(state) / gop.beta
    # average across particles
    nparticles = length(state.traces)
    magv = Vector{Float64}(undef, nparticles)
    ircv = Vector{Float64}(undef, nparticles)
    particles = sample_unweighted_traces(state, nparticles)
    @inbounds for i = 1:nparticles
        magv[i], ircv[i] = trace_mho(attx, attp, gop, particles[i])
    end

    mag = logsumexp(magv) - log(nparticles)
    irc = logsumexp(ircv) - log(nparticles)
    mho = mag - irc

    # print_granularity_schema(chain)
    # println(task_relevance(attx,
    #                        attp.partition,
    #                        state.traces[1],
    #                        attp.nns))
    # println("mho = $(round(mag; digits=2))(mag) - "*
    #     " $(round(irc;digits=2))(irc) = $(mho)")
    # @show lml
    # @show mho + lml
    # println("--------------")
    
    mho + lml
end

function trace_mho(attx::AdaptiveAux,
                   attp::AdaptiveComputation,
                   gop::MhoFitness,
                   trace)
    tr = task_relevance(attx,
                        attp.partition,
                        trace,
                        attp.nns)
    mag = logsumexp(tr)
    importance = softmax(tr, attp.itemp)
    # @show tr
    # @show importance
    # state = get_last_state(trace)
    # obj = object_from_idx(state, argmax(mag))
    # println(obj)
    # print_granularity_schema(trace)
    # println("type: $(typeof(obj))\n pos: $(get_pos(obj)) \n vel: $(get_vel(obj))")
    c = irr_complexity(importance,
                       gop.complexity_factor,
                       gop.complexity_mass)
    (mag, c)
end

function irr_complexity(imp::Vector, factor::Float64, mass::Float64)
    n = length(imp)
    waste = 0.0
    @inbounds for i = 1:n
        w = mass * (1 - imp[i])^(factor)
        # println("i $(imp[i]) -> w $(w)")
        waste += w
    end
    # pad, will be denominator
    (waste + 1E-4)
end

function print_granularity_schema(chain::APChain)
    tr = retrieve_map(chain)
    print_granularity_schema(tr)
end

function print_granularity_schema(tr::InertiaTrace)
    print_granularity_schema(get_last_state(tr))
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
