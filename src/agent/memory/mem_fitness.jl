export (MLLFitness, MhoFitness)

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
    result = Vector{Float64}(undef, nparticles)
    @inbounds for i = 1:nparticles
        result[i] = trace_mho(attx, attp, gop, state.traces[i])
    end
    eff = logsumexp(result) - log(nparticles)
    mho = eff + lml
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
    trace_mho(mag, importance, gop.complexity_mass, gop.complexity_mass)
end

function trace_mho(mag, imp, factor, mass)
    n = length(imp)
    waste = 0.0
    for i = 1:n
        waste += exp(-factor * imp[i])
    end
    complexity = exp(mass * waste)
    return mag - log(complexity)
end
