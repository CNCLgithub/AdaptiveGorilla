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
        mstate.objectives[i] = logsumexp(mstate.objectives[i], v)
        # print_granularity_schema(vstate.chains[i])
        # println("℧: $(mstate.objectives[i])")
    end
    mstate.steps += 1
    return nothing
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


function trace_value(attx, attp, gop, trace)
    tr = task_relevance(attx,
                        attp.partition,
                        trace,
                        attp.nns)
    # @show tr
    mag = logsumexp(tr)
    importance = softmax(tr, attp.itemp)
    # @show importance
    trace_mho(mag, importance, 10.0, 2.0)
    # # rmul!(importance, 1.0 / maximum(importance))
    # # value = l2log(tr) # l2log(tr)
    # print_granularity_schema(trace)
    # @show l2log(tr)
    # cost = log(gop.size_cost * length(tr))
    # # -sum(abs.(importance .- mean(importance))) - cost
    # l2log(tr) - cost
end

function granularity_objective(ag::MentalModule{G},
                               ac::MentalModule{A},
                               chain::APChain,
    ) where {G<:AdaptiveGranularity, A<:AdaptiveComputation}
    gop, _ = mparse(ag)
    attp, attx = mparse(ac)
    @unpack partition, nns = attp
    @unpack state = chain
    # average across particles
    nparticles = length(state.traces)
    result = Vector{Float64}(undef, nparticles)
    @inbounds for i = 1:nparticles
        result[i] = trace_value(attx, attp, gop, state.traces[i])
    end
    eff = logsumexp(result) - log(nparticles)
    lml = log_ml_estimate(state) / 5
    mho = gop.shift ? eff + lml : lml
    # print_granularity_schema(chain)
    # println("\t℧=$(mho)")
    # println("\teff=$(eff)")
    # println("\tlml=$(lml)")
    # mho
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

    # Repopulate and shift granularity
    memstate.objectives .-= log(memstate.steps)
    ws = softmax(memstate.objectives, memp.tau)
    # println("℧: $(memstate.objectives)")
    # @show ws
    metric = attp.map_metric
    next_gen = Vector{Int}(undef, visp.h)
    Distributions.rand!(Distributions.Categorical(ws), next_gen)
    # if memp.shift
    #     Distributions.rand!(Distributions.Categorical(ws), next_gen)
    # else
    #     next_gen[:] .= 1:visp.h
    # end
    for i = 1:visp.h
        parent = visstate.chains[next_gen[i]] # PFChain
        template = retrieve_map(parent) # InertiaTrace
        tr = task_relevance(attx,
                            attp.partition,
                            template,
                            attp.nns)
        cm = memp.shift ?
            shift_granularity(template, tr, metric) : noshift(template)
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


function noshift(t::InertiaTrace)
    cm = choicemap()
    cm[:s0 => :nsm] = 1 # no change
    return cm
end

function shift_granularity(t::InertiaTrace, tre::Vector{Float64}, metric::PreMetric)
    cm = choicemap()
    if rand() > 0.5
        sample_granularity_move!(cm, t, tre, metric)
    else
        cm[:s0 => :nsm] = 1 # no change
    end
    return cm
end

function sample_granularity_move!(cm::Gen.ChoiceMap, t::InertiaTrace,
                                  ws::Vector{Float64},
                                  metric::PreMetric)

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
            # Scale by similarity
            pairs[i] = logsumexp(importance[a], importance[b])
            # pairs[i] = logsumexp(importance[a], importance[b]) +
            #     log(dissimilarity(t, metric, a, b))
            # pairs[i] = log(1.01 - (importance[a] + importance[b])) -
            #     log(dissimilarity(t, metric, a, b))
        end
        # Greedy optimization
        # pairs = softmax(pairs, 0.001) #TODO: hyper parameter
        # @show pairs
        # pair_idx = categorical(pairs)
        min_w = minimum(pairs)
        pair_idx = rand(findall(==(min_w), pairs))
        selected = combination(ntotal, 2, pair_idx)
        # @show selected
        # @show dissimilarity(t, metric, selected...)
        # @show pairs[pair_idx]
        # @show dissimilarity(t, metric, 5,8)
        # @show pairs[comb_index(ntotal, [5, 8])]
        cm[:s0 => :nsm] = 3 # merge branch
        cm[:s0 => :state => :pair] = pair_idx
        # error()
    end
    return nothing
end

