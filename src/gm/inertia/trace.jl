################################################################################
# Trace methods
################################################################################

gen_fn(::InertiaWM) = wm_inertia
const InertiaIR = Gen.get_ir(wm_inertia)
const InertiaTrace = Gen.get_trace_type(wm_inertia)

function extract_rfs_subtrace(trace::InertiaTrace, t::Int64)
    # StaticIR names and nodes
    outer_ir = Gen.get_ir(wm_inertia)
    kernel_node = outer_ir.call_nodes[2] # :kernel
    kernel_field = Gen.get_subtrace_fieldname(kernel_node)
    # subtrace for each time step
    kernel_traces = getproperty(trace, kernel_field)
    sub_trace = kernel_traces.subtraces[t] # :kernel => t
    # StaticIR for `inertia_kernel`
    kernel_ir = Gen.get_ir(inertia_kernel)
    xs_node = kernel_ir.call_nodes[5] # :xs
    xs_field = Gen.get_subtrace_fieldname(xs_node)
    # `RFSTrace` for :masks
    getproperty(sub_trace, xs_field)
end


"""
    $(TYPEDSIGNATURES)

Posterior probability that gorilla is detected.
---
Criterion:
    1. A gorilla is present
    2. There are 5 singles
    3. All targets (xs[1-4]) are tracked
    4. The gorilla (x = n + 1) is tracked
"""
function detect_gorilla(trace::InertiaTrace,
                        ntargets::Int = 4,
                        nobj::Int = 8,
                        temp::Float64 = 1.0)

    t = first(get_args(trace))
    t == 0 && return 0.0 # TODO: fix issue with `reinit_chain`
    rfs = extract_rfs_subtrace(trace, t)
    pt = rfs.ptensor
    scores = rfs.pscores
    nx,ne,np = size(pt)
    state = get_last_state(trace)
    # No gorilla visable
    if (nx != nobj + 1 )
        return 0.0
    end
    ns = length(state.singles)
    if ns == 0
        # No individuals to detect gorilla
        return 0.0
    end
    # No birth individual
    if (object_count(state) != nobj + 1 )
        return 0.0
    end
    result = -Inf
    @inbounds for p = 1:np
        for s = 1:ns
            pt[nx, ns, p] || continue
            result = logsumexp(result, scores[p])
        end
    end
    result -= rfs.score
    return exp(result)
end

function had_birth(trace::InertiaTrace,
                   nobj::Int = 8)
    state = get_last_state(trace)
    total = object_count(state)
    total > nobj
end

function get_last_state(tr::InertiaTrace)
    t, wm, istate = get_args(tr)
    t == 0 ? istate : last(get_retval(tr))
end
