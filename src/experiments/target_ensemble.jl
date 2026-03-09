export TEnsExp

#################################################################################
# Constructors
#################################################################################

"""
$(TYPEDEF)

The Target-Ensemble experiment (study 2).

Each trial has a sister that is identical in terms of motion, but has all colors
swapped (including the gorilla).

In one case, only one of the targets bounces against the wall, rendering the rest
not relevant for the task. The gorilla may appear on either the lone target, or
on one of the grouped targets.

In the other case, the targets are matched for the number of bounces but are not
otherwise restricted in terms of how many of them can bounce off walls. In this
case, the gorilla appears on the same objects as the previous condition, with
all of the parent objects being not relevant to the same degree.

The targets are always Light objects.
"""
struct TEnsExp <: Experiment
    "Number of time steps"
    frames::Int64
    "Whether the parent is lone or grouped object"
    lone_parent::Bool
    "Whether to swap colors (included gorilla)"
    swap_colors::Bool
    "Observations for model"
    observations::Vector{ChoiceMap}
    "Query to initialize percept"
    init_query::IncrementalQuery
end

"""
$(TYPEDSIGNATURES)

---

Initializes a trial for the Target-Ensemble experiment.

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
function TEnsExp(dpath::String, wm::InertiaWM, trial_idx::Int64,
                 swap_color::Bool,
                 lone_parent::Bool,
                 frames::Int64;
                 show_gorilla::Bool = true)
    ws, obs = load_tens_trial(wm, dpath, trial_idx, lone_parent,
                              swap_color; frames=frames,
                              show_gorilla=show_gorilla)

    gm = gen_fn(wm)
    args = (0, wm, ws) # t = 0
    # uniform initial granularity
    init_cm = choicemap()
    init_cm[:s0 => :nsm] = 1
    # argdiffs: only `t` changes
    q = IncrementalQuery(gm, init_cm, args, INERTIA_ARG_DIFFS, 1)
    TEnsExp(frames, lone_parent, swap_color, obs, q)
end

#################################################################################
# API
#################################################################################

function get_map_hyper(agent::Agent)
    memp, memstate = mparse(agent.memory)
    visp, visstate = mparse(agent.perception)
    
    idx = argmax(memstate.objectives)
    best_hp = visstate.chains[idx]
    retrieve_map(best_hp)
end

function run_analyses(::TEnsExp, agent::Agent)
    # gorilla_p = exp(detect_gorilla(get_map_hyper(agent)))
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
function test_agent!(agent::Agent, exp::TEnsExp, stepid::Int)
    obs = get_obs(exp, stepid)
    agent_step!(agent, stepid, obs)
    run_analyses(exp, agent)
end


#################################################################################
# Helpers
#################################################################################

function get_obs(exp::TEnsExp, idx::Int64)
    exp.observations[idx]
end

"""
$(TYPEDSIGNATURES)

Loads scene `i` from the dataset, and generates observations.

`gorilla_threshold` determines when the gorilla observation will first and last
appear, based on threshold of separation (e.g., 50% of the gorilla is unoccluded
by the parent)
"""
function load_tens_trial(wm::WorldModel,
                         dpath::String,
                         trial_idx::Int,
                         lone_parent::Bool,
                         swap_color::Bool;
                         frames::Int = 10,
                         show_gorilla = true,
                         gorilla_threshold::Float64 = 0.25)
    raw_length = 0
    open(dpath, "r") do io
        manifest = JSON3.read(io)["manifest"]
        raw_length = manifest["frames"]
    end

    trial_length = min(raw_length, frames)
    # Variables
    local gorilla, positions
    observations = Vector{ChoiceMap}(undef, trial_length - 1)
    open(dpath, "r") do io
        # loading a vec of positions
        data = JSON3.read(io)["trials"][trial_idx]
        gorilla = data["gorilla"]
        positions = data["positions"]
    end
    gorilla_dur = min(raw_length - gorilla["frame"], 64)
    gorilla_color = swap_color ? Dark : Light
    parent = lone_parent ? 4 : 1
    # first frame gets GT
    istate = initial_state(wm, positions[1], 4, swap_color)
    for t = 2:trial_length
        cm = choicemap()
        # Observations associated with each object
        step = positions[t]
        nobj = length(step)
        @inbounds for i = 1:nobj
            xy = step[i]
            mat = xor(swap_color, i <= 4) ? Light : Dark
            write_obs_mask!(
                cm, wm, t, i, xy, mat;
                prefix = (t, i) -> i,
            )
        end
        if show_gorilla
            # Gorilla observations: orbits around parent
            write_tens_gorilla!(cm, t, gorilla["frame"],
                                gorilla_dur, gorilla_threshold,
                                wm.single_size, step[parent],
                                wm, gorilla_color, nobj + 1)
        end
        observations[t-1] = cm
    end
    (istate, observations)
end

function write_tens_gorilla!(
    cm::ChoiceMap,
    t::Int64,
    start::Int,
    gorilla_dur,
    occ_thresh,
    size,
    loc,
    wm::InertiaWM,
    color,
    idx,
    )
    dt = t - start
    (0 > dt || dt > gorilla_dur) && return nothing
    gx, gy = loc
    angle = ( 2 * dt * pi ) / gorilla_dur
    radius_pct = sin(0.5 * angle)
    # eg., only show for frames with at least 1/2 unoccluded
    radius_pct < occ_thresh && return nothing
    radius = radius_pct * 8 * size
    xy = orbit_position(gx, gy, radius, angle)
    write_obs_mask!(
        cm, wm, t, idx, xy, color;
        prefix = (t, i) -> i,
    )
    return nothing
end

function orbit_position(
    x::Real,
    y::Real,
    r::Real,
    a::Real,
    )
    # Calculate the child's position using parametric equations of a circle
    x = x + r * cos(a)
    y = y + r * sin(a)
    return S2V(x, y)
end



function count_collisions(exp::TEnsExp)
    (_, wm, _) = exp.init_query.args
    nframes = length(exp.observations)
    r = wm.single_size
    h = wm.area_height
    w = wm.area_width
    col_frame = fill(-Inf, Int64(wm.object_rate))
    count = 0
    for t = 1:nframes
        detections = to_array(get_obs(exp, t), Detection)
        count, col_frame = count_collisions!(count, col_frame, t, detections, r, h, w)
    end
    return count
end

function render_frame(exp::TEnsExp, t::Int, objp = ObjectPainter())
    obs = to_array(get_obs(exp, t), Detection)
    MOTCore.paint(objp, obs)
    return nothing
end

function render_agent_state(exp::TEnsExp, agent::Agent, t::Int, path::String)
    objp = ObjectPainter()
    idp = IDPainter()


    init = InitPainter(path = "$(path)/$(t).png",
                       background = "#808080")

    _, wm, _ = exp.init_query.args
    # setup
    MOTCore.paint(init, wm)
    # inferred states
    render_frame(agent.perception, agent.attention, agent.memory, objp)
    # render_frame(agent.planning, t)
    # observations
    render_frame(exp, t, objp)
    finish()
    return nothing
end
