export MemoryModule,
    assess_granularity!,
    regranularize!,
    AdaptiveGranularity,
    GranularityEstimates


"""
Defines the granularity mapping for a current trace format
"""
@with_kw struct AdaptiveGranularity <: MemoryProtocol
    "Temperature for chain resampling"
    tau::Float64 = 1.0
    shift::Bool = true
    size_cost::Float64 = 10.0
end

mutable struct GranularityEstimates <: MentalState{AdaptiveGranularity}
    objectives::Vector{Float64}
    steps::Int
end

function GranularityEstimates(n::Int)
    GranularityEstimates(zeros(n), 0)
end

# TODO: document
function MemoryModule(p::AdaptiveGranularity, n::Int)
    MentalModule(p, GranularityEstimates(n))
end

function print_granularity_schema(chain::APChain)
    tr = retrieve_map(chain)
    state = get_last_state(tr)
    ns = length(state.singles)
    ne = length(state.ensembles)
    println("MAP Granularity: $(ns) singles; $(ne) ensembles")
    ndark = count(x -> material(x) == Dark, state.singles)
    println("\tSingles: $(ndark) Dark | $(ns-ndark) Light")
    return nothing
end

function assess_granularity!(
    mem::MentalModule{M},
    att::MentalModule{A},
    vis::MentalModule{V}
    ) where {M<:AdaptiveGranularity,
             A<:AdaptiveComputation,
             V<:HyperFilter}
    mp, mstate = mparse(mem)
    hf, vstate = mparse(vis)
    # REVIEW: why does adding the line below
    # always lead to the last hparticle dominating?
    # update_task_relevance!(att)
    for i = 1:hf.h
        v = granularity_objective(mem, att, vstate.chains[i])
        # print_granularity_schema(vstate.chains[i])
        mstate.objectives[i] = logsumexp(mstate.objectives[i], v)
    end
    # @show mstate.objectives
    # @show softmax(mstate.objectives, mp.tau)
    mstate.steps += 1
    return nothing
end

function granularity_objective(ag::MentalModule{G},
                               ac::MentalModule{A},
                               chain::APChain,
    ) where {G<:AdaptiveGranularity, A<:AdaptiveComputation}
    gop, _ = mparse(ag)
    attp, attx = mparse(ac)
    @unpack partition, nns = attp
    @unpack state = chain
    trace = retrieve_map(chain)
    tr = task_relevance(attx,
                        attp.partition,
                        trace,
                        attp.nns)
    len = length(tr)
    mag = l2log(tr) - (log(gop.size_cost * len))
    # lml = log_ml_estimate(state)
    # @show mag
    # @show lml
    # lml + mag
end

function regranularize!(mem::MentalModule{M},
                        att::MentalModule{A},
                        vis::MentalModule{V},
                        t::Int
    ) where {M<:AdaptiveGranularity,
             A<:AdaptiveComputation,
             V<:HyperFilter}

    # Resampling weights
    memp, memstate = mparse(mem)
    visp, visstate = mparse(vis)
    attp, attx = mparse(att)
    # not time yet
    (t > 0 && t % visp.dt != 0) && return nothing

    ws = softmax(memstate.objectives, memp.tau)
    # Repopulate and shift granularity
    next_gen = Vector{Int}(undef, visp.h)
    Distributions.rand!(Distributions.Categorical(ws), next_gen)
    for i = 1:visp.h
        # parent = visstate.chains[i] # PFChain
        parent = visstate.chains[next_gen[i]] # PFChain
        template = retrieve_map(parent) # InertiaTrace
        tr = task_relevance(attx,
                            attp.partition,
                            template,
                            attp.nns)
        cm = memp.shift ?
            shift_granularity(template, tr) : noshift(template)
        visstate.new_chains[i] = reinit_chain(parent, template, cm)
    end

    visstate.age = 1 # TODO: 0 or 1?
    temp_chains = visstate.chains
    visstate.chains = visstate.new_chains
    visstate.new_chains = temp_chains

    fill!(memstate.objectives, -Inf)
    memstate.steps = 0

    return nothing
end


function noshift(t::InertiaTrace)
    cm = choicemap()
    cm[:s0 => :nsm] = 1 # no change
    return cm
end

function shift_granularity(t::InertiaTrace, tre::Vector{Float64})
    _, wm, _ = get_args(t)
    state = get_last_state(t)
    @unpack singles, ensembles = state
    cm = choicemap()
    if rand() > 0.5
        ns = length(singles)
        ne = length(ensembles)
        sample_granularity_move!(cm, tre, ns, ne)
    else
        cm[:s0 => :nsm] = 1 # no change
    end
    return cm
end

function sample_granularity_move!(cm::Gen.ChoiceMap, ws::Vector{Float64},
                                  nsingle::Int, nensemble::Int)
    ntotal = nsingle + nensemble
    nsplit = nensemble
    nmerges = ncr(ntotal, 2)
    # REVIEW: does this promote merging?
    split_prob = nsplit / (nsplit + nmerges)
    move = if rand() < split_prob
        # sample which ensemble to split
        split_ws = softmax(ws[nsingle+1:end])
        cm[:s0 => :nsm] = 2 # split branch
        cm[:s0 => :state => :idx] = categorical(split_ws)
    else
        # sample which pair to merge
        npairs = ncr(ntotal, 2)
        pairs = Vector{Float64}(undef, npairs)
        # Coarse importance filter
        importance = softmax(ws, 2000.0) #TODO: hyper parameter
        for i = 1:npairs
            (a, b) = combination(ntotal, 2, i)
            # Pr(Merge) inv. prop. importance
            pairs[i] = 1.0 - (importance[a] + importance[b])
        end
        # Greedy optimization
        # pairs = softmax(pairs, 0.1) #TODO: hyper parameter
        # pair_idx = categorical(pairs)
        pair_idx = argmax(pairs)
        cm[:s0 => :nsm] = 3 # merge branch
        cm[:s0 => :state => :pair] = pair_idx
    end
    return nothing
end

