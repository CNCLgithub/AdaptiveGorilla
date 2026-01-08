export Experiment, get_obs, count_collisions,
    render_agent_state, run_analyses, test_agent!

abstract type Experiment end

"""
Runs the agent on the next tick of the experiment and
returns some measurements of the agent's interal state.

### Implementations:

$(METHODLIST)
"""
function test_agent! end


"""
Determines the number of collisions for a trial.

### Implementations:

$(METHODLIST)
"""
function count_collisions end

# TODO: document the other exports!

include("most.jl")
include("target_ensemble.jl")
include("load_curve.jl")
