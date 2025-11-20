export (MLLOptim, MhoOptim)

################################################################################
# Marginal Log-likelihood Optimization
################################################################################

"""
Optimizes marginal log-likelihood across hyper particles
"""
struct MLLOptim <: GranularityOptimizer
end

# TODO: what to do about arg 2?
function memory_objective(optim::MLLOptim,
                          chain::APChain)
    log_ml_estimate(chain.state)
end

################################################################################
# Granularity Optimization
################################################################################

"""
Defines the granularity mapping for a current trace format
"""
@with_kw struct MhoOptim <: GranularityOptimizer
    att::MentalModule{<:AdaptiveComputation}
    beta::Float64 = 5.0
    complexity_mass::Float64 = 10.0
    complexity_factor::Float64 = 2.0
end


function memory_objective(gop::MhoOptim,
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

function trace_mho(mag, imp, factor = 1.0, mass = 1.0)
    n = length(imp)
    waste = 0.0
    for i = 1:n
        waste += exp(-factor * imp[i])
    end
    complexity = exp(mass * waste)
    return mag - log(complexity)
end


function trace_mho(attx::AdaptiveAux, attp::AdaptiveComputation,
                   gop::MhoOptim, trace)
    tr = task_relevance(attx,
                        attp.partition,
                        trace,
                        attp.nns)
    mag = logsumexp(tr)
    importance = softmax(tr, attp.itemp)
    trace_mho(mag, importance, gop.complexity_mass, gop.complexity_mass)
end
