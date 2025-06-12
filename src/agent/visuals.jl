using Luxor: finish


function render_frame(perception::MentalModule{V},
                      attention::MentalModule{A},
                      objp = ObjectPainter()) where {V<:HyperFilter,
                                                     A<:AdaptiveComputation}
    vp, vs = mparse(perception)
    attp, attx = mparse(attention)
    for i = 1:vp.h
        chain = vs.chains[i]
        trace = retrieve_map(chain)
        state = get_last_state(trace)
        tr = task_relevance(attx,
                            attp.partition,
                            trace,
                            attp.nns)
        importance = softmax(tr, attp.itemp)
        MOTCore.paint(objp, state, importance)
    end
    return nothing
end

