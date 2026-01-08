## GEN TRACE HELPERS

function find_inf_scores(trace::Gen.Trace)
    choices = get_choices(trace)
    find_inf_scores(trace, choices, [])
end

function find_inf_scores(trace::Gen.Trace,
                         choices::Gen.ChoiceMap,
                         prefix::Vector)
    for (key, val) = get_values_shallow(choices)
        addr = foldr(=>, prefix; init = key)
        selection = select(addr)
        w = Gen.project(trace, selection)
        if isnan(w) || isinf(w)
            println("#####################")
            println("Offending address: $(repr(addr)):")
            println("---")
            println("Value : $(trace[addr])")
            println("---")
            println("Args  : $(get_args(trace)[2:end])")
            println("#####################")
            return nothing
        end
    end
    for (key, submap) = get_submaps_shallow(choices)
        _prefix = deepcopy(prefix)
        push!(_prefix, key)
        find_inf_scores(trace, submap, _prefix)
    end
    return nothing
end

# NOTE: I don't think this is correct;
# I needed this for debugging
function Gen.project(trace::GenRFS.RFSTrace,
                     selection::DynamicSelection)
    trace.score
end
    
