abstract type Experiment end

struct Gorillas <: Experiment
    gorilla_idx::Int64
    gorilla_color::Material
    frames::Int64
    observations::Vector{ChoiceMap}
end

function Gorllias(dpath::String, wm::InertiaWM, trial_idx::Int64,
                  gorilla_idx::Int64, gorilla_color::Material,
                  frames::Int64)
    obs = load_from_dataset(wm, dpath, trial_idx, frames,
                            gorilla_color, gorilla_idx)
    Gorillas(gorilla_idx, gorilla_color, frames, obs)
end

function get_obs(exp::Gorillas, idx::Int64)
    exp.observations[idx]
end

"""
$(TYPEDSIGNATURES)

Loads scene `i` from the dataset, and generates observations
"""
function load_from_dataset(wm::WorldModel, dpath::String, i::Int,
                           time_steps::Int = 10,
                           gorilla_color::Material = Dark,
                           gorilla_idx::Int = 9)
    trial_length = 0
    open(dpath, "r") do io
        manifest = JSON3.read(io)["manifest"]
        trial_length = manifest["duration"]
    end

    trial_length = min(trial_length, time_steps)

    observations = Vector{ChoiceMap}(undef, trial_length)
    open(dpath, "r") do io
        # loading a vec of positions
        data = JSON3.read(io)["trials"][i]
        for t = 1:trial_length
            cm = choicemap()
            step = data[t]["positions"]
            write_obs!(cm, wm, step, t, gorilla_color,
                       gorilla_idx)
            observations[t] = cm
        end
    end

    return observations
end
