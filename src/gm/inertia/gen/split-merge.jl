# @gen function informed_split_merge(st::InertiaState)
#     ws = cohesion_weights(st)
#     selected ~ categorical(Fill(1.0 / nobj), nobj)
#     smws = informed_split_merge_weigths(st, selected)
#     switch_idx ~ categorical(smws)
#     new_st ~ split_merge_switch(switch_idx, st, selected)
#     return new_st
# end


# REVIEW: Where is this used?
@gen function naive_split_merge(st::InertiaState)
    nobj = count_objects(st)
    selected ~ categorical(Fill(1.0 / nobj), nobj)
    smws = split_merge_weigths(st, selected)
    switch_idx ~ categorical(smws)
    new_st ~ split_merge_switch(switch_idx, st, selected)
end

# Assumes object `idx` is an ensemble
@gen function split_object(st::InertiaState, idx::Int)
    # split according to the prior
    e = st.ensembles[idx]
    n = round(Int64, rate(e))
    objects ~ recurse_prior(ProdNode(0.5, n, wm), 1)
    new_st::InertiaState = replace_objects(st, idx, objects)
    return new_st
end

# assumes there are at least 2 objects
@gen static function merge_object(st::InertiaState, idx::Int)
    # select other object at random
    c = count_objects(st) - 1
    other ~ categorical(Fill(1.0 / c, c))
    new_st::InertiaState = merge_objects(st, idx, other)
    return new_st
end

# Does nothing
@gen static function no_sm(st::InertiaState, idx::Int)
    return st
end

const split_merge_switch(no_sm, split_object, merge_object)

@gen function split_merge_proposal(tr::InertiaTrace)
    # which objects are doing poorly at predicting detections
    ws = inv_goodness_fits(tr) # TODO: Implement me!
    idx = :selected ~ categorical(ws)
    # guess whether split or merge is better
    move_ws = repair_weights(tr, idx)
    sidx = :switch_idx ~ categorical(move_ws)
    new_st ~ sm_proposal_switch(sidx, tr, idx)
    return new_st
end

const sm_proposal_switch = Switch(
    no_sm,
    split_proposal, # which bit should be split off?
    merge_proposal  # which objects should be merged?
)


function enumerate_mergers(st::InertiaState)
    @unpack singles, ensembles = st
    (isempty(singles) && length(ensembles) == 1) && return [1.0]
    # TODO: implement
end

function merge_objects(st::InertiaState, idx::Int)
    idx == 1 && return st
    # TODO: implement
end

@gen function merge_pairs(state)
    ws = enumerate_mergers(state)
    idx ~ categorical(ws) # 1 reserved for no merge
    merged = merge_objects(state, idx)
    return (pair_idx, merged)
end


@gen function split_object(state)
    ws = enumerate_splits(state)
    idx ~ categorical(ws) # 1 reserved for no split
    split = split_object(state, idx)
end
