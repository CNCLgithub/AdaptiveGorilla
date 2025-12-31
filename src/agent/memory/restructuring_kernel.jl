export StaticRKernel, SplitMergeKernel,
    SplitMergeHeuristic, UniformSplitMerge, MhoSplitMerge

using SparseArrays: sparsevec

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
        # SPLIT | MERGE
        if rand() < split_prob(kappa.heuristic, t)
            sample_split_move!(cm, kappa.heuristic, t)
        else
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
    # Only one ensemble -> can't merge
    representation_count(tr) == 1 && return 1.0
    # No ensemble -> can't split
    ensemble_count(tr) == 0 && return 0.0
    # 50/50 split/merge
    return 0.5
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
    "Reference to and AdaptiveComputation module"
    att::MentalModule{<:AdaptiveComputation}
    "Maximum element count considered for merging"
    max_merge_elems::Int64 = 5
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
    if isempty(ws)
        @show single_count(t)
        @show ensemble_count(t)
    end

    max_w = maximum(ws)
    pair_idx = rand(findall(==(max_w), ws))
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

    # Determine importance: Less importance -> higher merge weight
    # NOTE: importance temperature scales with |tr|
    #   - at low |tr|, differences don't matter as much
    temp = attp.itemp - logsumexp(tr)
    temp = max(temp, 1.0)
    importance = softmax(tr, temp)

    # The weight of each merge pair is simply the sum of their importance values
    ntotal = length(importance)
    # Only consider the `k` least important elements
    # (to reduce combinatoric explosions)
    ncandidates = min(ntotal, h.max_merge_elems)
    cand_indices = partialsortperm(importance, 1:ncandidates)
    sparse_pairs = ncr(ncandidates, 2)
    pair_id = Vector{Int64}(undef, sparse_pairs)
    pair_ws = Vector{Float64}(undef, sparse_pairs)
    for i = 1:sparse_pairs
        (x, y) = combination(ncandidates, 2, i)
        a = cand_indices[x]
        b = cand_indices[y]
        # Pr(Merge) inv. prop. importance
        pair_id[i] = comb_index(ntotal, [a, b])
        pair_ws[i] = - (importance[a] + importance[b])
    end
    # Normalize
    rmul!(pair_ws, 1.0 / sum(pair_ws))
    
    # Store merge-weights in a sparse vector
    total_pairs = ncr(ntotal, 2)
    ws = sparsevec(pair_id, pair_ws, total_pairs)

    # NOTE: to retrieve members use:
    # selected = combination(total_pairs, 2, pair_idx)
    return ws
end
