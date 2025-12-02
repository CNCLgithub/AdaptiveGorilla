function single_ancestral_proposal(trace::InertiaTrace,
                                   single::Int)
    # println("single ancestral proposal")
    # need to see if idx corresponds to:
    # 1. gorilla
    # 2. normal individual
    t, wm = get_args(trace)
    addr = :kernel => t => :forces => single
    # addr = if single <= wm.object_rate
    #     :kernel => t => :forces => single
    # else
    #     :kernel => t => :birth => :birth
    # end
    new_trace, w, _ = regenerate(trace, select(addr))

    if isnan(w)
        state = trace[:kernel => t]
        new_state = new_trace[:kernel => t]
        @show (length(state.singles), length(state.ensembles))
        @show (length(new_state.singles), length(new_state.ensembles))
        @show single
        @show addr
        @show get_score(trace)
        @show project(trace, select(:init_state => :n))
        @show project(trace, select(addr))
        @show project(trace, select(:kernel => t => :forces))
        @show project(trace, select(:kernel => t => :eshifts))
        @show project(trace, select(:kernel => t => :merge => :mergers))
        @show project(trace, select(:kernel => t => :split => :splits => :splits))
        @show project(trace, select(:kernel => t => :birth))
        @show project(trace, select(:kernel => t => :xs))
        # @show get_score(new_trace)
        # @show project(trace, select(addr))
        # @show project(new_trace, select(addr))
        # @show project(trace, select(:xs))
        # @show project(new_trace, select(:xs))
        prev = get_submap(get_choices(trace), addr)
        new = get_submap(get_choices(new_trace), addr)
        # prev = get_choices(trace)
        # new = get_choices(new_trace)
        println("prev : $(sprint(show, "text/plain", prev))")
        println("new : $(sprint(show, "text/plain", new))")
        error("NaN proposal weight")
    end
    (new_trace, w)
end

function ensemble_ancestral_proposal(trace::InertiaTrace,
                                     idx::Int)
    t = first(get_args(trace))
    new_trace, w, _ = regenerate(trace, select(
        :kernel => t => :eshifts => idx => :force,
        :kernel => t => :eshifts => idx => :spread,
    ))
    (new_trace, w)
end

function baby_ancestral_proposal(trace::InertiaTrace)
    t = first(get_args(trace))
    selection = select(
        :kernel => t => :bd => :i,
        :kernel => t => :bd => :switch,
    )
    new_trace, w, discard = regenerate(trace, selection)
    # orig_choices = get_selected(get_choices(trace), selection)
    # new_choices = get_selected(get_choices(new_trace), selection)
    # @show w
    # display(orig_choices)
    # display(new_choices)
    (new_trace, w)
end

"""
Proposes new objects near location
"""
@gen function bd_loc_proposal(trace::InertiaTrace,
                              posx::Float64,
                              posy::Float64,
                              sigma::Float64)
    t, wm = get_args(trace)
    # 50% unless death occurred (then 0)
    w = trace[:kernel => t => :bd => :i] == 3 ? 0.0 : 0.9
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
    posx, posy = get_pos(object_from_idx(trace, idx))
    object = object_from_idx(trace, idx)
    sigma = if typeof(object) <: InertiaSingle
        3.0 * wm.single_size
    else
        3.0 * get_var(object)
    end
    (posx, posy, sigma)
end

function bd_loc_transform(trace::InertiaTrace, idx::Int64)
    trans = SymmetricTraceTranslator(bd_loc_proposal,
                                     bd_loc_args(trace, idx),
                                     bd_loc_involution)
    new_trace, w = apply_translator(trans, trace)
end


@gen function nearby_single(wm::InertiaWM, px::Float64, py::Float64, sigma = 100.0)
    x ~ normal(px, sigma)
    y ~ normal(py, sigma)
    ms = materials(wm)
    nm = length(ms)
    mws = Fill(1.0 / nm, nm)
    material ~ categorical(mws)
    ang ~ uniform(0.0 , 2.0 * pi)
    std ~ normal(wm.vel, 1.0)
    loc = S2V(x, y)
    vel = S2V(std*cos(ang), std*sin(ang))
    result::InertiaSingle = InertiaSingle(ms[material], loc, vel)
    return result
end

"""
Proposes new objects near poorly explained detections
"""
@gen function birth_death_ddp(trace::InertiaTrace)
    t, wm = get_args(trace)
    # negative marginal log likelihood
    # NOTE: required outside of if statement
    # to statisfy involution; otherwise
    # cannot recover u''
    nmls = marginal_ll(trace)
    lmul!(-1.0, nmls)
    # @show nmls
    # if length(nmls) == 9
    #     error()
    # end
    # sample from worst explained detections
    didx ~ categorical(softmax(nmls))
    # birth ~ bernoulli(birth_weight(wm, state))
    # 50% unless death occurred (then 0)
    w = trace[:kernel => t => :bd => :i] == 3 ? 0.0 : 0.5
    birth ~ bernoulli(w)
    if birth
        # make proposal around detection
        detection = trace[:kernel => t => :xs => didx]
        px, py = position(detection)
        baby ~ nearby_single(wm, px, py)
        force ~ inertia_force(wm, baby)
    end

    return single_count(trace)
end

@transform birth_death_involution (tr, u) to (tprime, uprime) begin

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
        @copy(u[:didx], uprime[:didx])

    elseif tbd == 2 && !ubirth  # T' + U' -> T, U
        obj_idx = nsingles
        # T' (1)
        @write(tprime[:kernel => t => :bd => :i], 1, :discrete)
        # U' (4)
        @write(uprime[:birth], true, :discrete)
        @copy(tr[:kernel => t => :bd => :switch => :birth],
              uprime[:baby])
        @copy(tr[:kernel => t => :forces => obj_idx], uprime[:force])
        @copy(u[:didx], uprime[:didx])
        # @copy(u[:baby], uprime[:baby])
        # @copy(u[:force], uprime[:force])
    end

    # Swapping births
    if tbd == 2  && ubirth
        obj_idx = nsingles
        @copy(u[:baby], tprime[:kernel => t => :bd => :switch => :birth])
        @copy(u[:force], tprime[:kernel => t => :forces => obj_idx])
        @copy(tr[:kernel => t => :bd => :switch => :birth], uprime[:baby])
        @copy(tr[:kernel => t => :forces => obj_idx], uprime[:force])
        @copy(u[:didx], uprime[:didx])
        @copy(u[:birth], uprime[:birth])
    end

    # No births
    if tbd == 1 && !ubirth
        # Do nothing =)
        @copy(tr[:kernel => t => :bd => :i],
              tprime[:kernel => t => :bd => :i])
        @copy(u[:didx], uprime[:didx])
        @copy(u[:birth], uprime[:birth])
    end


    # Deaths
    # In case of death, proposal does not sample
    if tbd == 3
        # Do nothing =)
        @copy(tr[:kernel => t => :bd],
              tprime[:kernel => t => :bd])
        @copy(u[:didx], uprime[:didx])
        @copy(u[:birth], uprime[:birth])
    end
end

function birth_death_transform(trace::Gen.Trace)
    trans = SymmetricTraceTranslator(birth_death_ddp,
                                     (),
                                     birth_death_involution)
    new_trace, w = apply_translator(trans, trace)
    # new_trace, w = trans(trace, check = true)
end

function apply_translator(translator, prev_model_trace)
    check = false
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

    if check
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
        println("u'")
        display(get_choices(backward_proposal_trace))
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
