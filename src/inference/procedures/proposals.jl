struct ReweightBlock
    proposal
    steps::Int32
end

function (b::ReweightBlock)(state::Gen.ParticleFilterState,
                            pidx::Int,
                            args...)
    @unpack proposal, steps = b
    s = state.traces[pidx]
    c = 0
    for _ = 1:steps
        s_prime, w = proposal(s, args...)
        if log(rand()) < w # MH acceptance ratio
            s = s_prime
            # mh reweighting
            state.log_weights[pidx] += w
            c += 1
        end
    end
    # println("Acceptance ratio: $(c / steps)")
    state.traces[pidx] = s
    return nothing
end

function single_ancestral_proposal(trace::InertiaTrace,
                                   single::Int)
    println("single ancestral proposal")
    # need to see if idx corresponds to:
    # 1. gorilla
    # 2. normal individual
    t, wm = get_args(trace)
    addr = if single <= wm.object_rate
        :kernel => t => :forces => single
    else
        :kernel => t => :birth => :birth
    end
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
    new_trace, w, _ = regenerate(trace, select(
        :kernel => t => :birth => :pregnant
    ))
    (new_trace, w)
end

@gen function nearby_single(wm::InertiaWM, px::Float64, py::Float64)
    x ~ normal(px, 100.0)
    y ~ normal(py, 100.0)
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

# TODO: make involution
@gen function baby_local_proposal(trace::InertiaTrace, parent::Int)
    t, wm = get_args(trace)
    state = trace[:kernel => t]
    birthed = trace[:kernel => t => :birth => :pregnant]
    s_or_l ~ bernoulli(0.5)
    if xor(birthed, s_or_l)
        px, py = get_pos(state.singles[parent])
        baby ~ nearby_single(wm, px, py)
    end
    # if birthed
    #     if s_or_l
    #         # no baby =(
    #     else
    #         x, y = get_pos(state.singles[parent])
    #         bx ~ normal(x, 100.0)
    #         by ~ normal(y, 100.0)
    #     end
    # else
    #     if s_or_l
    #         x, y = get_pos(state.singles[parent])
    #         bx ~ normal(x, 100.0)
    #         by ~ normal(y, 100.0)
    #     else
    #         # no changes
    #     end
    # end
    return s_or_l
end

@transform baby_local_involution (tr, u) to (tprime, uprime) begin

    t = first(get_args(tr))

    s_or_l = @read(u[:s_or_l], :discrete)
    birthed = @read(tr[:kernel => t => :birth => :pregnant], :discrete)

    if birthed
        if s_or_l # Structure change (baby -> no baby)
            # no baby =(
            println("Birthed + Struct : baby -> no baby")
            @write(tprime[:kernel => t => :birth => :pregnant], false, :discrete)
            @write(uprime[:s_or_l], true, :discrete) # !birthed, !s_or_l => no change
            @copy(tr[:kernel => t => :birth => :birth],
                  uprime[:baby])
            # @copy(tr[:kernel => t => :forces => idx + 1],
            #       uprime[:force])
        else # local change (pos -> pos)
            # going forward
            println("Birthed + Local : pos -> pos'")
            @copy(u[:baby], tprime[:kernel => t => :birth => :birth])
            # @copy(u[:force], tprime[:kernel => t => :forces => idx])
            # going back
            @write(uprime[:s_or_l], false, :discrete)
            @copy(tr[:kernel => t => :birth => :birth],
                uprime[:baby])
            # @copy(tr[:kernel => t => :forces => idx],
            #       uprime[:force])
        end
    else
        if s_or_l # Structure change (no baby -> baby)
            # st = @read(tr[:kernel => t], :discrete)
            # idx = length(st.singles)
            println("NoBirth + Struct : no baby -> baby")
            @write(tprime[:kernel => t => :birth => :pregnant], true, :discrete)
            @copy(u[:baby], tprime[:kernel => t => :birth => :birth])
            # @copy(u[:force], tprime[:kernel => t => :forces => idx])

            @write(uprime[:s_or_l], true, :discrete) # baby -> no baby =(
            # @copy(tr[:kernel => t => :forces => idx], uprime[:force])
        else # local change
            println("NoBirth + Local : nothing -> nothing")
            # nothing happens here =|
            @write(uprime[:s_or_l], false, :discrete)
        end
    end
end

function baby_local_transform(trace::Gen.Trace, idx::Int)
    t = first(get_args(trace))
    display(get_submap(get_choices(trace), :kernel => t => :birth))
    trans = SymmetricTraceTranslator(baby_local_proposal,
                                     (idx,),
                                     baby_local_involution)
    new_trace, w = apply_translator(trans, trace)
    # new_trace, w = trans(trace, check = true)
end

function apply_translator(translator, prev_model_trace)
    check = true
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
        println("fwd")
        display(get_choices(forward_proposal_trace))
        println("bwd")
        display(get_choices(backward_proposal_trace))
        println("fwd-rt")
        display(get_choices(forward_proposal_trace_rt))
        println("tr")
        display(get_submap(get_choices(prev_model_trace),
                           :kernel => t => :birth))
        println("tprime")
        display(get_submap(get_choices(new_model_trace),
                           :kernel => t => :birth))
        println("tr-rt")
        display(get_submap(get_choices(prev_model_trace_rt),
                           :kernel => t => :birth))
        Gen.check_round_trip(prev_model_trace, prev_model_trace_rt,
                         forward_proposal_trace, forward_proposal_trace_rt)
    end

    @show new_model_score
    @show prev_model_score
    @show backward_proposal_score
    @show forward_proposal_score
    @show log_abs_determinant
    @show log_weight
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
