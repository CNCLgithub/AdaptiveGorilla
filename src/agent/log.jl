export query_from_dataset,
    render_inference

function digest_obs(chain::APChain)
    particles = chain.state
    best_idx = argmax(particles.log_weights)
    best_particle = particles.traces[best_idx]
    t, wm = get_args(best_particle)
    best_particle[:kernel => t => :xs]
end

function digest_auxillary(chain::APChain)
    @unpack partition  = chain.proc.attention
    particles = chain.state
    best_idx = argmax(particles.log_weights)
    best_particle = particles.traces[best_idx]
    importance(partition, best_particle, chain.auxillary.tr)
end

function digest_map_state(chain::APChain)
    particles = chain.state
    best_idx = argmax(particles.log_weights)
    best_particle = particles.traces[best_idx]
    _, wm = get_args(best_particle)
    _, states = get_retval(best_particle)
    state = last(states)
end

function digest_single_positions(chain::APChain)
    np = length(chain.state.traces)
    traces = sample_unweighted_traces(chain.state, np)
    result = Vector{Vector{S2V}}(undef, np)
    @inbounds for i = 1:np
        (_, states) = Gen.get_retval(traces[i])
        current_state = last(states)
        @unpack singles = current_state
        k = length(singles)
        presult = Vector{S2V}(undef, k)
        for j = 1:k
            presult[j] = singles[j].pos
        end
       result[i] = presult
    end
    return result
end

function digest_gorilla(chain::APChain)
    np = length(chain.state.traces)
    traces = sample_unweighted_traces(chain.state, np)
    pgorilla = -Inf
    for i = 1:np
        pgorilla = logsumexp(pgorilla, detect_gorilla(traces[i]))
    end
    pgorilla -= log(np)
    @show pgorilla
    return pgorilla
end

"""
$(TYPEDSIGNATURES)

Loads scene `i` from the dataset, and generates observations
"""
function query_from_dataset(wm::WorldModel, dpath::String, i::Int,
                            time_steps::Int = 10,
                            gorilla_color::Material = Dark,
                            gorilla_idx::Int = 9)
    trial_length = 0
    open(dpath, "r") do io
        manifest = JSON3.read(io)["manifest"]
        trial_length = manifest["duration"]
    end

    trial_length = min(trial_length, time_steps + 1)
    @show trial_length

    # we will use the first time step to initialize the chain
    constraints = choicemap()
    # the rest will be observations
    observations = Vector{ChoiceMap}(undef, trial_length - 1)
    open(dpath, "r") do io
        # loading a vec of positions
        data = JSON3.read(io)["trials"][i]
        for t = 1:trial_length
            cm = choicemap()
            step = data[t]["positions"]

            if t == 1
                write_initial_constraints!(constraints, wm, step)
            else
                write_obs!(cm, wm, step, t - 1, gorilla_color,
                           gorilla_idx)
                observations[t - 1] = cm
            end
        end
    end

    k = trial_length - 1
    init_args = (0, wm)
    args = [(t, wm) for t in 1:k]
    argdiffs = Fill((UnknownChange(), NoChange()), k)

    latent_map = LatentMap(
        :aux=> digest_auxillary,
        # :wall_counts => digest_td_accuracy, # TODO: implement
        :map_state => digest_map_state,
        :xs => digest_obs,
        :pgorilla => digest_gorilla
    )

    Gen_Compose.SequentialQuery(latent_map,
                                wm_inertia,
                                init_args,
                                constraints,
                                args,
                                argdiffs,
                                observations)
end


using Luxor: finish

function render_inference(wm::InertiaWM, logger::ChainLogger,
                          path::String)
    bfr = buffer(logger)
    n = length(bfr)

    # Initialize some of the painters
    objp = ObjectPainter()
    idp = IDPainter()

    for i = 1:n
        step = bfr[i]
        state = step[:map_state]
        aux = step[:aux]

        init = InitPainter(path = "$(path)/$(i).png",
                           background = "white")

        obs = step[:xs]
        # Apply the painters
        paint(init, wm, state)
        paint(objp, state)
        paint(idp, state)
        paint(objp, obs)
        paint(state, aux)
        finish()
    end
    return nothing
end
