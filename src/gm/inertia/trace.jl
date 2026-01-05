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
    xs_node = kernel_ir.call_nodes[4] # :xs
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
function detect_gorilla(trace::InertiaTrace)
    t, wm, _ = get_args(trace)
    nobj = Int64(wm.object_rate)
    t == 0 && return -Inf
    state = get_last_state(trace)
    rfs = extract_rfs_subtrace(trace, t)
    nx,ne,np = size(rfs.ptensor)
    ns = length(state.singles)
    # Cases for 0 prob
    # No gorilla
    # No individuals to detect gorilla
    # No birth
    if (nx != nobj + 1 ) ||
        ns == 0  ||
        (object_count(state) != nobj + 1 )
        return -Inf
    end
    result = -Inf
    @inbounds for p = 1:np, e = 1:ns
        rfs.ptensor[nx, e, p] || continue
        result = logsumexp(result, rfs.pscores[p])
    end
    result - rfs.score
end

function had_birth_bool(trace::InertiaTrace)
    _, wm, _ = get_args(trace)
    nobj = Int64(wm.object_rate)
    state = get_last_state(trace)
    object_count(state) > nobj 
end

function had_birth(trace::InertiaTrace)
    had_birth_bool(trace) ? 0.0 : -Inf
end

function get_last_state(tr::InertiaTrace)
    t, wm, istate = get_args(tr)
    t == 0 ? istate : last(get_retval(tr))
end

function single_count(tr::InertiaTrace)
    state = get_last_state(tr)
    length(state.singles)
end

function ensemble_sum(tr::InertiaTrace)
    state = get_last_state(tr)
    sum(rate, state.ensembles; init=0.0)
end


function ensemble_count(tr::InertiaTrace)
    state = get_last_state(tr)
    length(state.ensembles)
end

# REVIEW: This is the number of objects representend.
# This is not the number of object representations
function object_count(tr::InertiaTrace)
    state = get_last_state(tr)
    object_count(state)
end

function representation_count(tr::InertiaTrace)
    state = get_last_state(tr)
    length(state.singles) + length(state.ensembles)
end

function marginal_ll(trace::InertiaTrace)
    # REVIEW: what about t=0?
    t = first(get_args(trace))
    rfs = extract_rfs_subtrace(trace, t)
    xs = Gen.to_array(rfs.choices, Detection)
    es = rfs.args[1]
    ml = GenRFS.support_table(es, xs)
    nx,ne,np = size(rfs.ptensor)
    result = fill(-Inf, nx)
    @inbounds for p = 1:np
        pmass = rfs.pscores[p] - rfs.score
        for e = 1:ne, x = 1:nx
            rfs.ptensor[x, e, p] || continue
            w = ml[e, x] + pmass
            result[x] = logsumexp(result[x], w)
        end
    end
    return result
end


function object_from_idx(tr::InertiaTrace, idx::Int64)
    state = get_last_state(tr)
    object_from_idx(state, idx)
end
