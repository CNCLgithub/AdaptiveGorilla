export render_agent_state


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
        paint(objp, state, importance)
    end
    return nothing
end

function render_agent_state(exp::Gorillas, agent::Agent, t::Int, path::String)
    objp = ObjectPainter()
    idp = IDPainter()


    init = InitPainter(path = "$(path)/$(t).png",
                       background = "white")

    _, wm, _ = exp.init_query.args
    # setup
    paint(init, wm)
    # observations
    render_frame(exp, t, objp)
    # inferred states
    render_frame(agent.perception, agent.attention, objp)
    render_frame(agent.planning, t)
    finish()
    return nothing
end
