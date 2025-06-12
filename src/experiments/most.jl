# TODO: exports
# export...

struct MostExp <: Experiment
    "Number of time steps"
    frames::Int64
    "Gorilla appearance"
    gorilla_color::Material
    "Observations for model"
    observations::Vector{ChoiceMap}
    "Query to initialize percept"
    init_query::IncrementalQuery
end

function MostExp(dpath::String, wm::InertiaWM,
                 trial_idx::Int64, gorilla_color::Material,
                 frames::Int64)
    ws, obs = load_most_trial(wm, dpath, trial_idx, frames,
                              gorilla_color)

    gm = gen_fn(wm)
    args = (0, wm, ws) # t = 0
    # argdiffs: only `t` changes
    argdiffs = (Gen.UnknownChange(), Gen.NoChange(), Gen.NoChange())
    # uniform initial granularity
    init_cm = choicemap()
    init_cm[:s0 => :nsm] = 1
    q = IncrementalQuery(gm, init_cm, args, argdiffs, 1)
    MostExp(frames, gorilla_color, obs, q)
end

function get_obs(exp::MostExp, idx::Int64)
    exp.observations[idx]
end

"""
$(TYPEDSIGNATURES)

Loads scene `i` from the dataset, and generates observations
"""
function load_most_trial(wm::WorldModel,
                         dpath::String,
                         trial_idx::Int,
                         time_steps::Int = 10,
                         gorilla_color::Material = Dark)
    trial_length = 0
    open(dpath, "r") do io
        manifest = JSON3.read(io)["manifest"]
        trial_length = manifest["duration"]
    end

    trial_length = min(trial_length, time_steps)
    # Variables
    local istate::InertiaState, gorilla, positions
    observations = Vector{ChoiceMap}(undef, trial_length - 1)
    open(dpath, "r") do io
        # loading a vec of positions
        data = JSON3.read(io)["trials"][trial_idx]
        gorilla = data["gorilla"]
        positions = data["positions"]
    end
    # first frame gets GT
    istate = initial_state(wm, data["positions"][1])
    for t = 2:trial_length
        cm = choicemap()
        # Observations associated with each object
        step = positions[t]
        nobj = length(step)
        @inbounds for i = 1:nobj
            xy = step[i]
            mat = i <= 4 ? Light : Dark
            write_obs_mask!(
                cm, wm, t, i, xy, mat;
                prefix = (t, i) -> i,
            )
        end
        # Gorilla observations
        # moves right to left
        delta_t = t - gorilla["frame"]
        if delta_t > 0
            x0 = 0.5 * wm.area_width
            x = x0 - delta_t * gorilla["speedx"]
            write_obs_mask!(
                cm, wm, t, i, S2V(x, 0.0), gorilla_color;
                prefix = (t, i) -> i,
            )
        end
        observations[t-1] = cm
    end

    (istate, observations)
end

function render_frame(exp::MostExp, t::Int, objp = ObjectPainter())
    obs = to_array(get_obs(exp, t), Detection)
    paint(objp, obs)
    return nothing
end

function collision_expectation(exp::MostExp)
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
