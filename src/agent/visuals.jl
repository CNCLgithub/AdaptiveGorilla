using Luxor: finish
using MOTCore: _draw_line

function render_assigments(trace::InertiaTrace)
    t = first(get_args(trace))
    # Regranularization occurs at the end of a time step.
    # Therefor, t=0 until the next observation is fed in.
    # Here, we lose a frame of rendering.
    t == 0 && return nothing
    rfs = extract_rfs_subtrace(trace, t)
    xs = Gen.to_array(rfs.choices, Detection)
    state = get_last_state(trace)

    nx,ne,np = size(rfs.ptensor)
    porder = sortperm(rfs.pscores; rev = true)
    pmass = -Inf
    pidx = 1
    while pmass < log(0.95)
        p = porder[pidx]
        pmass = logsumexp(pmass, rfs.pscores[p] - rfs.score)
        @inbounds for e = 1:(ne-1) # last elem is catchall
            epos = get_pos(object_from_idx(state, e))
            for x = 1:nx
                rfs.ptensor[x, e, p] || continue
                xpos = position(xs[x])
                _draw_line(epos, xpos, "blue"; opacity=0.25)
            end
        end
    end
    return nothing
end

function render_frame(perception::MentalModule{V},
                      attention::MentalModule{A},
                      memory::MentalModule{G},
                      objp = ObjectPainter()
                      ) where {V<:HyperFilter,
                               A<:AdaptiveComputation,
                               G<:AdaptiveGranularity}
    vp, vs = mparse(perception)
    attp, attx = mparse(attention)
    memp, memx = mparse(memory)

    # Get best hyper particle
    chain = vs.chains[1]
    mho = -Inf
    for i = 1:vp.h
        _chain = vs.chains[i]
        _mho = granularity_objective(memory, attention, _chain)
        if _mho > mho
            chain = _chain
        end
    end

    trace = retrieve_map(chain)
    state = get_last_state(trace)
    tr = task_relevance(attx,
                        attp.partition,
                        trace,
                        attp.nns)
    importance = softmax(tr, attp.itemp)
    MOTCore.paint(objp, trace, importance)
    # render_assigments(trace)
    return nothing
end

