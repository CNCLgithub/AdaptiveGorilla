export LoadCurve

#################################################################################
# Constructors
#################################################################################

struct LoadCurve <: Experiment
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
function LoadCurve(wm::InertiaWM, nlight::Int64, ndark::Int64,
                   frames::Int64)
    ws, obs = gen_trial(wm, nlight, ndark, frames)

    gm = gen_fn(wm)
    args = (0, wm, ws) # t = 0
    # argdiffs: only `t` changes
    argdiffs = (Gen.UnknownChange(), Gen.NoChange(), Gen.NoChange())
    # uniform initial granularity
    init_cm = choicemap()
    init_cm[:s0 => :nsm] = 1
    q = IncrementalQuery(gm, init_cm, args, argdiffs, 1)
    LoadCurve(frames, Dark, obs, q)
end

#################################################################################
# API
#################################################################################

function run_analyses(experiment::LoadCurve, agent::Agent)
    gorilla_p = exp(estimate_marginal(agent.perception,
                                  detect_gorilla, ()))
    birth_p = exp(estimate_marginal(agent.perception,
                                  had_birth, ()))
    col_p = planner_expectation(agent.planning)
    Dict(:gorilla_p => gorilla_p,
         :collision_p => col_p, #abs(col_p - col_gt) / col_gt,
         :birth_p => birth_p)
end

"""
$(TYPEDSIGNATURES)

Runs the agent on the next frame of the trial and
returns a `Dict` containing:

- the probability that the agent noticed the gorilla
- The agent's expectation over the number of event counts
"""
function step_agent!(agent::Agent, exp::LoadCurve, stepid::Int)
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

function gen_trial(wm::InertiaWM, nlight::Int64, ndark::Int64, nframes::Int64)
    dgp = init_wm(nlight + ndark)
    _, states = wm_scholl(nframes, dgp)
    gorilla = Dict(
        # onset time
        :frame =>
            uniform_discrete(round(Int64, 0.15 * nframes),
                             round(Int64, 0.25 * nframes)),
        # speed to move across
        :speedx => normal(dgp.vel, 1.0),
    )
    # first frame gets GT
    istate = initial_state(wm, states[1], nlight)
    observations = Vector{ChoiceMap}(undef, nframes - 1)
    @inbounds for j in 2:nframes
        cm = choicemap()
        state = states[j]
        objects = state.objects
        no = length(objects)
        step = Vector{S2V}(undef, no)
        for k in 1:no
            obj = objects[k]
            xy = obj.pos
            mat = k <= nlight ? Light : Dark
            write_obs_mask!(
                cm, wm, j, k, xy, mat;
                prefix = (t, i) -> i,
            )
        end
        # Gorilla observations
        # moves right to left
        delta_t = j - gorilla[:frame]
        if delta_t > 0
            x0 = 0.5 * wm.area_width
            x = x0 - delta_t * gorilla[:speedx]
            write_obs_mask!(
                cm, wm, j, no+1, S2V(x, 0.0), Dark;
                prefix = (t, i) -> i,
            )
        end
        observations[j-1] = cm
    end
    (istate, observations)
end

function init_wm(n_dots::Int)
    SchollWM(
        ;
        n_dots = n_dots,
        dot_radius = 5.0,
        area_width = 720.0,
        area_height = 480.0,
        vel=4.5,
        vel_min = 3.5,
        vel_max = 5.5,
        vel_step = 0.20,
        vel_prob = 0.20
    )
end

function get_obs(exp::LoadCurve, idx::Int64)
    exp.observations[idx]
end

function render_frame(exp::LoadCurve, t::Int, objp = ObjectPainter())
    obs = to_array(get_obs(exp, t), Detection)
    MOTCore.paint(objp, obs)
    return nothing
end

function count_collisions(exp::LoadCurve)
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

# TODO: part of API instead of helper?
function render_agent_state(exp::LoadCurve, agent::Agent, t::Int, path::String)
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
