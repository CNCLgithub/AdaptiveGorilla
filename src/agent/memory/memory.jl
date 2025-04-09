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
    println("Chain has: $(ns) singles; $(ne) ensembles")
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
    end
    # @show mstate.objectives
    mstate.steps += 1
    return nothing
end



function granularity_objective(ag::MentalModule{G},
                               ac::MentalModule{A},
                               chain::APChain,
    ) where {G<:AdaptiveGranularity, A<:AdaptiveComputation}
    attp, attx = mparse(ac)
    @unpack partition, nns = attp
    @unpack state = chain
    trace = retrieve_map(chain)
    tr = task_relevance(attx,
                        attp.partition,
                        trace,
                        attp.nns)
    len = length(tr)
    mag = l2log(tr) - (log(len))
    # Running averages
    # mag = -Inf
    # @inbounds for i = 1:np
    #     tr = task_relevance(attx,
    #                         attp.partition,
    #                         state.traces[i],
    #                         attp.nns)
    #     @show tr
    #     len = length(tr)
    #     _mag = l2log(tr)
    #     # @show _mag
    #     mag = logsumexp(mag, _mag - log(len^2))
    # end
    # mag -= log(np)
    # mag
    # lml = log_ml_estimate(state)
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
    @show memstate.objectives
    @show ws
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
        cm = shift_granularity(template, tr)
        visstate.new_chains[i] = reinit_chain(parent, template, cm)
        # visstate.new_chains[i] = reinit_chain(parent, template)
    end

    visstate.age = 1 # TODO: 0 or 1?
    temp_chains = visstate.chains
    visstate.chains = visstate.new_chains
    visstate.new_chains = temp_chains

    fill!(memstate.objectives, -Inf)
    memstate.steps = 0

    return nothing
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

function sample_granularity_move!(cm::Gen.ChoiceMap, nsingle::Int, nensemble::Int)
    ntotal = nsingle + nensemble
    nmerges = ncr(ntotal, 2)
    split_prob = nensemble / (nensemble + nmerges)
    move = if rand() < split_prob
        # sample which ensemble to split
        cm[:s0 => :nsm] = 2
        cm[:s0 => :state => :idx] = rand(1:nensemble)
    else
        # sample which pair to merge
        cm[:s0 => :nsm] = 3
        cm[:s0 => :state => :pair] = rand(1:nmerges)
    end
    return nothing
end

function sample_granularity_move!(cm::Gen.ChoiceMap, ws::Vector{Float64},
                                  nsingle::Int, nensemble::Int)
    ntotal = nsingle + nensemble
    nmerges = ncr(ntotal, 2)
    split_prob = nensemble / (nensemble + nmerges)
    move = if rand() < split_prob
        # sample which ensemble to split
        split_ws = softmax(ws[nsingle+1:end])
        cm[:s0 => :nsm] = 2
        cm[:s0 => :state => :idx] = categorical(split_ws)
    else
        # sample which pair to merge
        npairs = ncr(ntotal, 2)
        pairs = Vector{Float64}(undef, npairs)
        for i = 1:npairs
            (a, b) = combination(ntotal, 2, i)
            # prevent log(-x) in next line
            w = min(0.0, logsumexp(ws[a], ws[b]))
            pairs[i] = log1mexp(w)
        end
        pairs = softmax(pairs, 0.001)
        cm[:s0 => :nsm] = 3
        cm[:s0 => :state => :pair] = categorical(pairs)
    end
    return nothing
end

function apply_granularity_move(m::MergeMove, wm::InertiaWM, state::InertiaState)
    @unpack singles, ensembles = state
    ns = length(singles)
    ne = length(ensembles)
    a = object_from_idx(state, m.a)
    b = object_from_idx(state, m.b)
    e = apply_merge(a, b)
    # need to determine what object types were merged
    delta_ns = 0
    delta_ns += isa(a, InertiaSingle) ? -1 : 0
    delta_ns += isa(b, InertiaSingle) ? -1 : 0
    delta_ne = 0
    if isa(a, InertiaSingle) && isa(b, InertiaSingle)
        delta_ne = 1

    elseif isa(a, InertiaEnsemble) && isa(b, InertiaEnsemble)
        delta_ne = -1
    end
    new_singles = Vector{InertiaSingle}(undef, ns + delta_ns)
    new_ensembles = Vector{InertiaEnsemble}(undef, ne + delta_ne)
    # Remove any merged singles
    c = 1
    for i = 1:ns
        (isa(a, InertiaSingle) && i == m.a) && continue
        (isa(b, InertiaSingle) && i == m.b) && continue
        new_singles[c] = singles[i]
        c += 1
    end
    # Remove merged ensembles
    c = 1
    for i = 1:ne
        ((isa(b, InertiaEnsemble) && i + ns == m.b)  ||
            (isa(a, InertiaEnsemble) && i + ns == m.a)) &&
            continue
        new_ensembles[c] = ensembles[i]
        c += 1
    end
    new_ensembles[c] = e
    InertiaState(new_singles, new_ensembles)
end

function apply_granularity_move(m::SplitMove, wm::InertiaWM, state::InertiaState,
                                splitted::InertiaSingle)
    @unpack singles, ensembles = state
    ns = length(singles)
    ne = length(ensembles)
    e = ensembles[m.x]
    new_e = apply_split(e, splitted)
    # if the ensemble is a pair, we get two individuals
    delta_ns = e.rate == 2 ? 2 : 1
    delta_ne = e.rate == 2 ? -1 : 0
    new_singles = Vector{InertiaSingle}(undef, ns + delta_ns)
    new_ensembles = Vector{InertiaEnsemble}(undef, ne + delta_ne)
    
    new_singles[1:ns] = singles
    if e.rate == 2 # E -> (S, S)
        new_singles[ns + 1] = collapse(new_e)
        new_singles[ns + 2] = splitted
        c = 1
        for i = 1:ne
            i == m.x && continue
            new_ensembles[c] = ensembles[i]
            c += 1
        end
    else # E -> (E', S)
        new_singles[ns + 1] = splitted
        new_ensembles[:] = ensembles[:]
        new_ensembles[m.x] = new_e
    end
    InertiaState(new_singles, new_ensembles)
end


