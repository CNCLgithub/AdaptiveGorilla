struct ReweightBlock
    proposal
    steps::Int32
end

function (b::ReweightBlock)(state::Gen.ParticleFilterState,
                            pidx::Int,
                            args...)
    @unpack proposal, steps = b
    s = state.traces[pidx]
    for _ = 1:steps
        s_prime, w = proposal(s, args...)
        if log(rand()) < w # MH acceptance ratio
            s = s_prime
            # mh reweighting
            state.log_weights[pidx] += w
        end
    end
    state.traces[pidx] = s
    return nothing
end

function single_ancestral_proposal(trace::InertiaTrace,
                                   single::Int)
    t = first(get_args(trace))
    new_trace, w, _ = regenerate(trace, select(
        :kernel => t => :forces => single
    ))
    (new_trace, w)
end

function ensemble_ancestral_proposal(trace::InertiaTrace)
    t = first(get_args(trace))
    new_trace, w, _ = regenerate(trace, select(
        :kernel => t => :eshift => :force,
        :kernel => t => :eshift => :spread,
    ))
    (new_trace, w)
end

function baby_ancestral_proposal(trace::InertiaTrace)
    t = first(get_args(trace))
    new_trace, w, _ = regenerate(trace, select(
        :kernel => t => :birth
    ))
    (new_trace, w)
end

function apply_random_walk(trace::Gen.Trace, proposal, proposal_args)
    model_args = get_args(trace)
    argdiffs = map((_) -> NoChange(), model_args)
    proposal_args_forward = (trace, proposal_args...,)
    (fwd_choices, fwd_weight, _) = propose(proposal, proposal_args_forward)
    (new_trace, weight, _, discard) = Gen.update(trace,
                                                 fwd_choices)
    proposal_args_backward = (new_trace, proposal_args...,)
    (bwd_weight, _) = Gen.assess(proposal, proposal_args_backward, discard)
    alpha = weight - fwd_weight + bwd_weight
    if isnan(alpha)
        @show fwd_weight
        @show weight
        @show bwd_weight
        display(discard)
        error("nan in proposal")
    end
    (new_trace, alpha)
end
