export StaticRKernel, SplitMergeKernel,
    SplitMergeHeuristic, UniformSplitMerge, MhoSplitMerge


################################################################################
# Dummy Static Kernel
################################################################################

"No restructuring"
struct StaticRKernel <: RestructuringKernel end

function restructure_kernel(::StaticRKernel, t::InertiaTrace)
    cm = choicemap()
    cm[:s0 => :nsm] = 1 # no change
    return cm
end

################################################################################
# Split-merge Kernel
################################################################################

"A heuristic function over which objects to split or merge"
abstract type SplitMergeHeuristic end


@with_kw struct SplitMergeKernel <: RestructuringKernel
    heuristic::SplitMergeHeuristic
    restructure_prob::Float64 = 0.5
end

function restructure_kernel(kappa::SplitMergeKernel,
                            t::InertiaTrace)
    cm = choicemap()
    if rand() < kappa.restructure_prob
        if rand() < split_prob(kappa.heuristic, t)
            # SPLIT
            sample_split_move!(cm, kappa.heuristic, t)
        else
            # MERGE
            sample_merge_move!(cm, kappa.heuristic, t)
        end
    else
        cm[:s0 => :nsm] = 1 # no change
    end
    return cm
end

################################################################################
# Split-merge Heuristics
################################################################################

function split_prob(h::SplitMergeHeuristic, tr::InertiaTrace)
    nsingle = single_count(tr)
    nensemble = ensemble_count(tr)
    ntotal = nsingle + nensemble
    nsplit = nensemble
    nmerges = ncr(ntotal, 2)
    split_prob = nsplit / (nsplit + nmerges)
end

struct UniformSplitMerge <: SplitMergeHeuristic end



function sample_split_move!(cm::ChoiceMap,
                            h::UniformSplitMerge,
                            t::InertiaTrace)
    nensemble = ensemble_count(t)
    cm[:s0 => :nsm] = 2 # split branch
    cm[:s0 => :state => :idx] = rand(1:nensemble)
    return nothing
end

function sample_merge_move!(cm::ChoiceMap,
                            h::UniformSplitMerge,
                            t::InertiaTrace)

    nsingle = single_count(t)
    nensemble = ensemble_count(t)
    ntotal = nsingle + nensemble
    nmerges = ncr(ntotal, 2)
    cm[:s0 => :nsm] = 3 # merge branch
    cm[:s0 => :state => :pair] = rand(1:nmerges)
    return nothing
end

@with_kw struct MhoSplitMerge <: SplitMergeHeuristic
    att::MentalModule{<:AdaptiveComputation}
end

# NOTE: Greedy - argmax
function sample_split_move!(cm::ChoiceMap,
                            h::MhoSplitMerge,
                            t::InertiaTrace)
    ws = split_weights(h, t)
    cm[:s0 => :nsm] = 2 # split branch
    cm[:s0 => :state => :idx] = argmax(ws)
end

function sample_merge_move!(cm::ChoiceMap,
                            h::MhoSplitMerge,
                            t::InertiaTrace)
    ws = merge_weights(h, t)
    # Greedy optimization
    min_w = minimum(ws)
    pair_idx = rand(findall(==(min_w), ws))
    cm[:s0 => :nsm] = 3 # merge branch
    cm[:s0 => :state => :pair] = pair_idx
    return nothing
end

function split_weights(h::MhoSplitMerge,
                       t::InertiaTrace)
    # Estimate task-relevance
    attp, attx = mparse(h.att)
    tr = task_relevance(attx,
                        attp.partition,
                        t,
                        attp.nns)
    # task-relevance of ensembles
    nsingle = single_count(t)
    tr[nsingle+1:end]
end

function merge_weights(h::MhoSplitMerge,
                       t::InertiaTrace)
    # Estimate task-relevance
    attp, attx = mparse(h.att)
    tr = task_relevance(attx,
                        attp.partition,
                        t,
                        attp.nns)
    # TODO: Remove softmax? and log?
    importance = log.(softmax(tr, 10.0)) #TODO: hyper parameter

    nsingle = single_count(t)
    nensemble = ensemble_count(t)
    ntotal = nsingle + nensemble
    # sample which pair to merge
    npairs = ncr(ntotal, 2)
    ws = Vector{Float64}(undef, npairs)
    # Coarse importance filter
    # @show importance
    for i = 1:npairs
        (a, b) = combination(ntotal, 2, i)
        # Pr(Merge) inv. prop. importance
        ws[i] = logsumexp(importance[a], importance[b])
    end

    # NOTE: to retrieve members use:
    # selected = combination(npairs, 2, pair_idx)
    return ws
end
