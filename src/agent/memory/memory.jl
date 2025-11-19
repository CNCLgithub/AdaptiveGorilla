export MemoryModule,
    assess_granularity!,
    regranularize!,
    GranOptim,
    GranularityEstimates

################################################################################
# Memory Protocols
################################################################################

@with_kw struct HyperMem <: MemoryProtocol
    "Temperature for chain resampling"
    tau::Float64 = 1.0
    optim::MemoryOptimizer = MLLOptim()
end

mutable struct MemoryAssessments <: MentalState{HyperMem}
    objectives::Vector{Float64}
    steps::Int
end

function MemoryAssessments(size::Int)
    MemoryAssessments(zeros(size), 0)
end

# TODO: document
function MemoryModule(p::HyperMem, size::Int)
    MentalModule(p, MemoryAssessments(size))
end

function assess_memory!(
    mem::MentalModule{M},
    att::MentalModule{A},
    vis::MentalModule{V}
    ) where {M<:GranOptim,
             A<:AttentionProtocol,
             V<:HyperFilter}
    mp, mstate = mparse(mem)
    hf, vstate = mparse(vis)
    for i = 1:hf.h
        increment = memory_objective(mem, att, vstate.chains[i])
        mstate.objectives[i] = logsumexp(mstate.objectives[i], increment)
    end
    mstate.steps += 1
    return nothing
end

function optimize_memory!(mem::MentalModule{M},
                          att::MentalModule{A},
                          vis::MentalModule{V},
                          t::Int
    ) where {M<:HyperMem,
             A<:AttentionProtocol,
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
        cm = restructure_kernel(memp.optim, att, template)
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

"No restructuring"
struct StaticRKernel end

function restructure_kernel(::StaticRKernel, t::InertiaTrace)
    cm = choicemap()
    cm[:s0 => :nsm] = 1 # no change
    return cm
end

@with_kw struct SplitMergeKernel
    heurisitic::SplitMergeHeuristic
    restructure_prob::Float64 = 0.5
end

struct SplitMergeWeights
    split_prob::Float64
    split_weights::Vector{Float64}
    merge_weights::Vector{Float64}
end

function restructure_kernel(kappa::SplitMergeKernel,
                            t::InertiaTrace)
    weights = kappa.heurisitic(t)
    cm = choicemap()
    if rand() > kappa.restructure_prob
        sample_granularity_move!(cm, t, tr, attp.map_metric)
    else
        cm[:s0 => :nsm] = 1 # no change
    end
    return cm
end

abstract type SplitMergeHeuristic end

struct UniformSplitMerge end

struct MhoSplitMerge
    att::MentalModule{<:AdaptiveComputation}
end

function split_prob(h::SplitMergeHeuristic, tr::InertiaTrace)
    nsingle = single_count(tr)
    nensemble = ensemble_count(tr)
    ntotal = nsingle + nensemble
    nsplit = nensemble
    nmerges = ncr(ntotal, 2)
    split_prob = nsplit / (nsplit + nmerges)
end

function sample_split_move!(cm::ChoiceMap,
                            h::SplitMergeHeuristic,
                            tr::InertiaTrace)

    split_ws = softmax(ws[nsingle+1:end])
    cm[:s0 => :nsm] = 2 # split branch
    cm[:s0 => :state => :idx] = argmax(split_ws)
end

function (h::MhoSplitMerge)(tr::Trace)
    attp, attx = mparse(h.att)
    tr = task_relevance(attx,
                        attp.partition,
                        t,
                        attp.nns)

    nsingle = single_count(t)
    nensemble = ensemble_count(t)
    ntotal = nsingle + nensemble
    nsplit = nensemble
    nmerges = ncr(ntotal, 2)
    # split_prob = nsplit > 0 && any(!isinf, ws[nsingle+1:end]) ? 1.0 : 0.0
    split_prob = nsplit / (nsplit + nmerges)
    if rand() < split_prob
        # sample which ensemble to split
        split_ws = softmax(ws[nsingle+1:end])
        cm[:s0 => :nsm] = 2 # split branch
        cm[:s0 => :state => :idx] = argmax(split_ws)
    else
        # sample which pair to merge
        npairs = ncr(ntotal, 2)
        pairs = Vector{Float64}(undef, npairs)
        # Coarse importance filter
        importance = log.(softmax(ws, 10.0)) #TODO: hyper parameter
        # @show importance
        for i = 1:npairs
            (a, b) = combination(ntotal, 2, i)
            # Pr(Merge) inv. prop. importance
            pairs[i] = logsumexp(importance[a], importance[b])
        end
        # Greedy optimization
        min_w = minimum(pairs)
        pair_idx = rand(findall(==(min_w), pairs))
        selected = combination(ntotal, 2, pair_idx)
        cm[:s0 => :nsm] = 3 # merge branch
        cm[:s0 => :state => :pair] = pair_idx
    end
end

################################################################################
# Memory Optimizers
################################################################################

"Defines the particular strategy for optimizing hyper chains"
abstract type MemoryOptimizer end


"""
    memory_objective(optim, att, vision)

Returns the fitness of a hyper particle.
"""
function memory_objective end

"""
    restructure_kernel(optim, att, template::Trace)

Returns a choicemap that alters the memory schema of a trace.
"""
function restructure_kernel end


################################################################################
# Marginal Log-likelihood Optimization
################################################################################

"""
Optimizes marginal log-likelihood across hyper particles
"""
@with_kw struct MLLOptim <: MemoryOptimizer
    restructure::Bool = false
end

# TODO: what to do about arg 2?
function memory_objective(optim::MLLOptim,
                          ac::MentalModule{A},
                          chain::APChain,
    ) where {A<:AttentionProtocol}
    log_ml_estimate(chain.state)
end

# TODO: complete!
function restructure_heuristic(optim::MLLOptim)
    # random moves?
    # minimize complexity?
end

################################################################################
# Granularity Optimization
################################################################################

"""
Defines the granularity mapping for a current trace format
"""
@with_kw struct GranOptim <: MemoryOptimizer
    beta::Float64 = 5.0
end


function memory_objective(ag::MentalModule{G},
                          ac::MentalModule{A},
                          chain::APChain,
    ) where {G<:MemoryOptimizer, A<:AttentionProtocol}
    gop, _ = mparse(ag)
    attp, attx = mparse(ac)
    @unpack state = chain
    lml = log_ml_estimate(state) / 5
    if !gop.shift
        return lml
    end
    # average across particles
    nparticles = length(state.traces)
    result = Vector{Float64}(undef, nparticles)
    @inbounds for i = 1:nparticles
        result[i] = trace_value(attx, attp, gop, state.traces[i])
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


function trace_value(attx::AdaptiveAux, attp::AdaptiveComputation,
                     gop::GranOptim, trace)
    tr = task_relevance(attx,
                        attp.partition,
                        trace,
                        attp.nns)
    mag = logsumexp(tr)
    importance = softmax(tr, attp.itemp)
    # TODO: Parametrize!
    trace_mho(mag, importance, 10.0, 2.0)
end


function heurisitic_split_merge!(cm::Gen.ChoiceMap,
                                 t::InertiaTrace,
                                 ws::SplitMergeWeights)

    nsingle = single_count(t)
    nensemble = ensemble_count(t)
    ntotal = nsingle + nensemble
    nsplit = nensemble
    nmerges = ncr(ntotal, 2)
    # split_prob = nsplit > 0 && any(!isinf, ws[nsingle+1:end]) ? 1.0 : 0.0
    split_prob = nsplit / (nsplit + nmerges)
    if rand() < split_prob
        # sample which ensemble to split
        split_ws = softmax(ws[nsingle+1:end])
        cm[:s0 => :nsm] = 2 # split branch
        cm[:s0 => :state => :idx] = argmax(split_ws)
    else
        # sample which pair to merge
        npairs = ncr(ntotal, 2)
        pairs = Vector{Float64}(undef, npairs)
        # Coarse importance filter
        importance = log.(softmax(ws, 10.0)) #TODO: hyper parameter
        # @show importance
        for i = 1:npairs
            (a, b) = combination(ntotal, 2, i)
            # Pr(Merge) inv. prop. importance
            pairs[i] = logsumexp(importance[a], importance[b])
        end
        # Greedy optimization
        min_w = minimum(pairs)
        pair_idx = rand(findall(==(min_w), pairs))
        selected = combination(ntotal, 2, pair_idx)
        cm[:s0 => :nsm] = 3 # merge branch
        cm[:s0 => :state => :pair] = pair_idx
    end
    return nothing
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
    println("MAP Granularity: $(ns) singles; $(ne) ensembles; $(c) total")
    ndark = count(x -> material(x) == Dark, state.singles)
    println("\tSingles: $(ndark) Dark | $(ns-ndark) Light")
    println("\tEnsembles: $(map(e -> (rate(e), e.matws[1]), state.ensembles))")
    return nothing
end
