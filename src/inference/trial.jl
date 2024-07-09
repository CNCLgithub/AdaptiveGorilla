export query_from_dataset

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

"""
$(TYPEDSIGNATURES)

Loads scene `i` from the dataset, and generates observations
"""
function query_from_dataset(wm::WorldModel, dpath::String, i::Int,
                            time_steps::Int = 10)
    trial_length = 0
    open("pilot.json", "r") do io
        manifest = JSON3.read(io)["manifest"]
        trial_length = manifest["duration"]
    end

    trial_length = min(trial_length, time_steps + 1)
    @show trial_length

    # we will use the first time step to initialize the chain
    constraints = choicemap()
    # the rest will be observations
    observations = Vector{ChoiceMap}(undef, trial_length - 1)
    open("pilot.json", "r") do io
        # loading a vec of positions
        data = JSON3.read(io)["trials"][i]
        for t = 1:trial_length
            cm = choicemap()
            step = data[t]["positions"]

            if t == 1
                write_initial_constraints!(constraints, wm, step)
            else
                write_obs!(cm, wm, step, t - 1)
                observations[t - 1] = cm
            end
        end
    end

    k = trial_length - 1
    init_args = (0, wm)
    args = [(t, wm) for t in 1:k]
    argdiffs = Fill((UnknownChange(), NoChange()), k)

    latent_map = LatentMap(
        # :auxillary => digest_auxillary,
        # :wall_counts => digest_td_accuracy, # TODO: implement
        :positions => digest_single_positions,
    )

    Gen_Compose.SequentialQuery(latent_map,
                                wm_inertia,
                                init_args,
                                constraints,
                                args,
                                argdiffs,
                                observations)
end
