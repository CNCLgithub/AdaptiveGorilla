export MostExp

#################################################################################
# Constructors
#################################################################################

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

"""
$(TYPEDSIGNATURES)

Initializes a trial for the Most et al (2001) experiment.

---

Arguments:
- `dpath`: Path to the JSON dataset
- `wm` : Inertia World model
- `trial_idx`: The trial index
- `gorilla_color`: Gorilla appearance
- `frames`: The number of frames to load.

---

# Dataset format defined in protobuff

```proto
syntax = "proto3";

message Dot {
  float x = 1;
  float y = 2;
}

message Gorilla {
  float frame = 1;
  float parent = 2;
  float speedx = 3;
  float speedy = 4;
}

message Probe {
  uint32 frame = 1;
  uint32 obj = 2;
}

message Step {
  repeated Dot dots = 1;
}

message Trial {
  repeated Step steps = 1;
  optional Gorilla gorilla = 2;
  repeated Probe probes = 3;
  optional uint32 disappear = 4;
}

message Manifest {
  uint32 fps = 1;
  unit32 duration = 2;
  uint32 ntrials = 3;
  uint32 nframes = 4;
}

message Dataset {
  repeated Trial trials = 1;
  optional Manifest = 2;
}
```
"""
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

#################################################################################
# API
#################################################################################

function run_analyses(::MostExp, agent::Agent)
    gorilla_p = exp(estimate_marginal(agent.perception,
                                  detect_gorilla, ()))
    birth_p = exp(estimate_marginal(agent.perception,
                                  had_birth, ()))
    col_p = planner_expectation(agent.planning)

    Dict(:gorilla_p => gorilla_p,
         :collision_p => col_p,
         :birth_p => birth_p)
end

"""
$(TYPEDSIGNATURES)

Runs the agent on the next frame of the trial and
returns a `Dict` containing:

- the probability that the agent noticed the gorilla
- The agent's expectation over the number of event counts
"""
function step_agent!(agent::Agent, exp::MostExp, stepid::Int)
    obs = get_obs(exp, stepid)
    perceive!(agent, obs, stepid)
    attend!(agent, stepid)
    plan!(agent, stepid)
    memory!(agent, stepid)
    run_analyses(exp, agent)
end

#################################################################################
# Helpers
#################################################################################

function get_obs(exp::MostExp, idx::Int64)
    exp.observations[idx]
end

function load_most_trial(wm::WorldModel,
                         dpath::String,
                         trial_idx::Int,
                         time_steps::Int = 10,
                         gorilla_color::Material = Dark)
    trial_length = 0
    open(dpath, "r") do io
        manifest = JSON3.read(io)["manifest"]
        trial_length = manifest["frames"]
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
    istate = initial_state(wm, positions[1])
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
                cm, wm, t, nobj+1, S2V(x, 0.0), gorilla_color;
                prefix = (t, i) -> i,
            )
        end
        observations[t-1] = cm
    end
    (istate, observations)
end

function render_frame(exp::MostExp, t::Int, objp = ObjectPainter())
    obs = to_array(get_obs(exp, t), Detection)
    MOTCore.paint(objp, obs)
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

# TODO: part of API instead of helper?
function render_agent_state(exp::MostExp, agent::Agent, t::Int, path::String)
    objp = ObjectPainter()
    idp = IDPainter()


    # Perception
    init = InitPainter(path = "$(path)/perception-$(t).png",
                       background = "white")

    _, wm, _ = exp.init_query.args
    # setup
    MOTCore.paint(init, wm)
    # observations
    render_frame(exp, t, objp)
    # inferred states
    render_frame(agent.perception, agent.attention, agent.memory, objp)
    render_frame(agent.planning, t)
    finish()


    # Attention
    init = InitPainter(path = "$(path)/attention-$(t).png",
                       background = "white")

    _, wm, _ = exp.init_query.args
    # setup
    MOTCore.paint(init, wm)
    render_attention(agent.attention)
    finish()
    return nothing
end
