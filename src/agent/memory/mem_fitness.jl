export MLLFitness, MhoFitness

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
    @inbounds for i = 1:nparticles
        magv[i], ircv[i] = trace_mho(attx, attp, gop, state.traces[i])
    end

    mag = logsumexp(magv) - log(nparticles)
    irc = logsumexp(ircv) - log(nparticles)
    mho = mag - irc
    # print_granularity_schema(chain)
    # println("mho = $(round(mag; digits=2))(mag) - $(round(irc;digits=2))(irc) = $(mho)")
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
    # state = get_last_state(trace)
    # obj = object_from_idx(state, argmax(mag))
    # println("pos: $(get_pos(obj)) \n vel: $(get_vel(obj))")
    # @show importance
    c = comp_complexity(importance,
                        gop.complexity_factor,
                        gop.complexity_mass)
    (mag, c)
end

function comp_complexity(imp, factor, mass)
    n = length(imp)
    waste = 0.0
    @inbounds for i = 1:n
        # waste += exp(-factor * imp[i])
        # w = -factor * log(max(imp[i], 0.001))
        # w = (1 - imp[i]) / (factor + (1 - factor)*imp[i])
        w = (1 - imp[i])^(factor)
        # println("i $(imp[i]) -> w $(w)")
        waste += w
    end
    (waste + 1E-4) / mass
end

function print_granularity_schema(chain::APChain)
    tr = retrieve_map(chain)
    print_granularity_schema(tr)
end

function print_granularity_schema(tr::InertiaTrace)
    state = get_last_state(tr)
    ns = length(state.singles)
    ne = length(state.ensembles)
    c = object_count(tr)
    println("Granularity: $(ns) singles; $(ne) ensembles; $(c) total")
    ndark = count(x -> material(x) == Dark, state.singles)
    println("\tSingles: $(ndark) Dark | $(ns-ndark) Light")
    println("\tEnsembles: $(map(e -> (rate(e), e.matws[1]), state.ensembles))")
    return nothing
end
