
################################################################################
# Ancestral Proposals
################################################################################

function single_ancestral_proposal(trace::InertiaTrace,
                                   single::Int)
    t, wm = get_args(trace)
    selection = select(:kernel => t => :forces => single)
    new_trace, w, _ = regenerate(trace, selection)

    if isinf(w) || isnan(w)
        find_inf_scores(new_trace)
        error("Invalid score from ancestral proposal")
    end
    (new_trace, w)
end

function ensemble_ancestral_proposal(trace::InertiaTrace,
                                     idx::Int)
    t = first(get_args(trace))
    selection = select(
        :kernel => t => :eshifts => idx,
    )
    # selection = select(
    #     :kernel => t => :eshifts => idx => :fx,
    #     :kernel => t => :eshifts => idx => :fy,
    #     :kernel => t => :eshifts => idx => :spread,
    # )
    new_trace, w, _ = regenerate(trace, selection)

    # println("BEFORE:")
    # display(get_selected(get_choices(trace), selection))

    # println("AFTER:")
    # display(get_selected(get_choices(new_trace), selection))

    # error()

    if isinf(w) || isnan(w)
        find_inf_scores(new_trace)
        error("Invalid score from ancestral proposal")
    end
    (new_trace, w)
end

function baby_ancestral_proposal(trace::InertiaTrace)
    t = first(get_args(trace))
    selection = select(
        :kernel => t => :bd => :i,
        :kernel => t => :bd => :switch,
    )
    new_trace, w, discard = regenerate(trace, selection)
    if isinf(w) || isnan(w)
        find_inf_scores(new_trace)
        error("Invalid score from ancestral proposal")
    end
    # orig_choices = get_selected(get_choices(trace), selection)
    # new_choices = get_selected(get_choices(new_trace), selection)
    # if new_choices[:kernel => t => :bd => :i] == 2
    #     @show w
    #     display(orig_choices)
    #     display(new_choices)
    # end
    (new_trace, w)
end


################################################################################
# Involutive Proposals
################################################################################

@gen function nearby_single(wm::InertiaWM, px::Float64, py::Float64,
                            sigma = 100.0)
    xb, yb = object_bounds(wm)
    x ~ trunc_norm(px, sigma, xb[1], xb[2])
    y ~ trunc_norm(py, sigma, yb[1], yb[2])
    ms = materials(wm)
    nm = length(ms)
    mws = Fill(1.0 / nm, nm)
    material ~ categorical(mws)
    ang ~ uniform(0.0 , 2.0 * pi)
    mag ~ normal(wm.vel, 1.0)
    loc = S2V(x, y)
    vel = S2V(mag*cos(ang), mag*sin(ang))
    result::InertiaSingle = InertiaSingle(ms[material], loc, vel)
    return result
end

"""
Proposes new objects near location
"""
@gen function bd_loc_proposal(trace::InertiaTrace,
                              posx::Float64,
                              posy::Float64,
                              sigma::Float64)
    t, wm = get_args(trace)
    sc = single_count(trace)
    # 50% unless:
    # 1. death occurred
    # w = trace[:kernel => t => :bd => :i] == 3 ? 0.0 : 0.5
    w = had_birth_bool(trace) ||
        trace[:kernel => t => :bd => :i] == 3 ?
        0.0 : 1.0

    # println("BD proposal birth weight: $(w)")
    birth ~ bernoulli(w)
    if birth
        # make proposal around posx, posy
        baby ~ nearby_single(wm, posx, posy, sigma)
        force ~ inertia_force(wm, baby)
    end
    return single_count(trace)
end

@transform bd_loc_involution (tr, u) to (tprime, uprime) begin

    t = first(get_args(tr))

    ubirth = @read(u[:birth], :discrete)
    # 1 => no change | 2 => birth | 3 => death
    tbd = @read(tr[:kernel => t => :bd => :i],
                :discrete)
    nsingles = @read(u[], :discrete)
    # Adding or removing birth
    if tbd == 1 && ubirth # U + T -> T', U'
        obj_idx = nsingles + 1
        # T' (3)
        @write(tprime[:kernel => t => :bd => :i], 2, :discrete)
        @copy(u[:baby], tprime[:kernel => t => :bd => :switch => :birth])
        @copy(u[:force], tprime[:kernel => t => :forces => obj_idx])
        # U' (2)
        @write(uprime[:birth], false, :discrete)

    elseif tbd == 2 && !ubirth  # T' + U' -> T, U
        obj_idx = nsingles
        # T' (1)
        @write(tprime[:kernel => t => :bd => :i], 1, :discrete)
        # U' (4)
        @write(uprime[:birth], true, :discrete)
        @copy(tr[:kernel => t => :bd => :switch => :birth],
              uprime[:baby])
        @copy(tr[:kernel => t => :forces => obj_idx], uprime[:force])
    end

    # Swapping births
    if tbd == 2  && ubirth
        obj_idx = nsingles
        @copy(u[:baby], tprime[:kernel => t => :bd => :switch => :birth])
        @copy(u[:force], tprime[:kernel => t => :forces => obj_idx])
        @copy(tr[:kernel => t => :bd => :switch => :birth], uprime[:baby])
        @copy(tr[:kernel => t => :forces => obj_idx], uprime[:force])
        @copy(u[:birth], uprime[:birth])
    end

    # No births
    if tbd == 1 && !ubirth
        # Do nothing =)
        @copy(tr[:kernel => t => :bd => :i],
              tprime[:kernel => t => :bd => :i])
        @copy(u[:birth], uprime[:birth])
    end


    # Deaths
    # In case of death, proposal does not sample
    if tbd == 3
        # Do nothing =)
        @copy(tr[:kernel => t => :bd],
              tprime[:kernel => t => :bd])
        @copy(u[:birth], uprime[:birth])
    end
end

function bd_loc_args(trace::InertiaTrace, idx::Int64)
    t, wm, _ = get_args(trace)
    object = object_from_idx(trace, idx)
    posx, posy = get_pos(object)
    sigma = isa(object, InertiaSingle) ? wm.single_size : get_var(object)
    sigma *= 3
    (posx, posy, sigma)
end

function bd_loc_transform(trace::InertiaTrace, idx::Int64)
    trans = SymmetricTraceTranslator(bd_loc_proposal,
                                     bd_loc_args(trace, idx),
                                     bd_loc_involution)
    new_trace, w = apply_translator(trans, trace)
    if isinf(w) || isnan(w)
        msg = "Birth-death localized proposal encountered invalid weight\n"
        error(msg)
    end
    (new_trace, w)
end

function apply_translator(translator, prev_model_trace, check = false)

    # simulate from auxiliary program
    forward_proposal_trace =
        simulate(translator.q, (prev_model_trace, translator.q_args...,))

    # apply trace transform
    (new_model_trace, backward_proposal_trace, log_abs_determinant) =
        Gen.run_transform(translator, prev_model_trace, forward_proposal_trace)

    # compute log weight
    prev_model_score = get_score(prev_model_trace)
    new_model_score = get_score(new_model_trace)
    forward_proposal_score = get_score(forward_proposal_trace)
    backward_proposal_score = get_score(backward_proposal_trace)
    log_weight = new_model_score - prev_model_score +
        backward_proposal_score - forward_proposal_score + log_abs_determinant

    if isinf(log_weight) || isnan(log_weight) || check
        t = first(get_args(prev_model_trace))
        Gen.check_observations(get_choices(new_model_trace), EmptyChoiceMap())
        (prev_model_trace_rt, forward_proposal_trace_rt, _) =
            Gen.run_transform(translator, new_model_trace, backward_proposal_trace)
        println("tr")
        display(get_submap(get_choices(prev_model_trace),
                           :kernel => t))
        println("u")
        display(get_choices(forward_proposal_trace))
        println("t'")
        display(get_submap(get_choices(new_model_trace),
                           :kernel => t))
        find_inf_scores(new_model_trace)
        println("u'")
        display(get_choices(backward_proposal_trace))
        find_inf_scores(backward_proposal_trace)
        println("t''")
        display(get_submap(get_choices(prev_model_trace_rt),
                           :kernel => t))
        println("u''")
        display(get_choices(forward_proposal_trace_rt))
        Gen.check_round_trip(prev_model_trace, prev_model_trace_rt,
                         forward_proposal_trace, forward_proposal_trace_rt)
        @show new_model_score
        @show prev_model_score
        @show backward_proposal_score
        @show forward_proposal_score
        @show log_abs_determinant
        @show log_weight
        error("Invalid log weight")
    end

    return (new_model_trace, log_weight)
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
