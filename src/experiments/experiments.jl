export Experiment, Gorillas, load_from_dataset, get_obs,
    render_agent_state, run_analyses

abstract type Experiment end

struct Gorillas <: Experiment
    gorilla_idx::Int64
    gorilla_color::Material
    frames::Int64
    observations::Vector{ChoiceMap}
    init_query::IncrementalQuery
end

function Gorillas(dpath::String, wm::InertiaWM, trial_idx::Int64,
                  gorilla_idx::Int64, gorilla_color::Material,
                  frames::Int64)
    ws, obs = load_from_dataset(wm, dpath, trial_idx, frames,
                            gorilla_color, gorilla_idx)

    gm = gen_fn(wm)
    args = (0, wm, ws) # t = 0
    # argdiffs: only `t` changes
    argdiffs = (Gen.UnknownChange(), Gen.NoChange(), Gen.NoChange())
    # uniform initial granularity
    init_cm = choicemap()
    init_cm[:s0 => :nsm] = 1
    q = IncrementalQuery(gm, init_cm, args, argdiffs, 1)
    Gorillas(gorilla_idx, gorilla_color, frames, obs, q)
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
    local istate::InertiaState
    observations = Vector{ChoiceMap}(undef, trial_length - 1)
    open(dpath, "r") do io
        # loading a vec of positions
        data = JSON3.read(io)["trials"][i]
        # first frame gets GT
        istate = initial_state(wm, data[1]["positions"])
        for t = 2:trial_length
            cm = choicemap()
            step = data[t]["positions"]
            write_obs!(cm, wm, step, t-1, gorilla_color,
                       gorilla_idx;
                       prefix = (t, i) -> i,
                       )
            observations[t-1] = cm
        end
    end

    (istate, observations)
end

function render_frame(exp::Gorillas, t::Int, objp = ObjectPainter())
    obs = to_array(get_obs(exp, t), Detection)
    paint(objp, obs)
    return nothing
end

function collision_expectation(exp::Gorillas)
    (_, wm, _) = exp.init_query.args
    n = length(exp.observations)
    e = 0.0
    for t = 1:n
        detections = to_array(get_obs(exp, t), Detection)
        for d = detections
            intensity(d) != 1.0 && continue
            for k = 1:4 # each wall
                _ep, _ = colprob_and_agrad(position(d), wm.walls[k])
                e += _ep
            end
        end
    end
    return e
end

function run_analyses end
